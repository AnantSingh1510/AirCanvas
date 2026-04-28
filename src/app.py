import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from engine.gesture_engine import GestureEngine
from engine.gl_renderer import GLRenderer
from engine.cube import Cube

GESTURE_COLORS = {
    "POINT":    (50,  220,  80),
    "ROTATE":   (255, 165,   0),
    "GRAB":     (60,  140, 255),
    "EXTRUDE":  (60,  200, 255),
    "SCALE":    (160, 160, 160),
    "DESELECT": (255,  80,  80),
    "IDLE":     (160, 160, 160),
    "TWO_MOVE":  (60,  200, 255),
    "TWO_SCALE": (235, 95, 235),
    "TWO_ROTATE": (255, 185, 70),
    "TWO_IDLE":  (160, 160, 160),
}
GESTURE_ICONS = {
    "POINT":    "index — select face",
    "ROTATE":   "2 fingers — rotate",
    "GRAB":     "fist — move object",
    "EXTRUDE":  "thumb+pinky — extrude face",
    "SCALE":    "pinch — add 2nd hand",
    "DESELECT": "3 fingers — deselect",
    "IDLE":     "open palm",
    "TWO_MOVE":  "2 fists — move object",
    "TWO_SCALE": "2 index/pinch — scale",
    "TWO_ROTATE": "2 peace signs — orbit",
    "TWO_IDLE":  "2 hands visible",
}

class AirCanvas3D:
    def __init__(self):
        pygame.init()
        self.display = (800, 600)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("AirCanvas 3D")

        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.75,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )
        self.hand_engines = {}
        self.renderer = GLRenderer()
        self.renderer.init_gl(*self.display)

        self.cube = Cube(size=0.5)
        self.renderer.cube = self.cube

        self.smooth_index = None
        self.smooth_wrist = None
        self.alpha = 0.4

        self.prev_smooth_index = None
        self.prev_smooth_wrist = None
        self.prev_pinch        = None
        self.prev_two_center   = None
        self.prev_two_distance = None
        self.prev_two_angle    = None
        self.prev_two_depth    = None
        self.prev_two_scale_distance = None

        self.extruding        = False
        self.extrude_face_idx = None
        self.prev_extrude_control = None

        self.clock = pygame.time.Clock()
        self.fps   = 0
        self.cam_texture = glGenTextures(1)
        self.show_camera_background = False
        self.hand_window_name = "AirCanvas Hand View"

        cv2.namedWindow(self.hand_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.hand_window_name, 480, 360)

    def finger_to_world(self, index_pos):
        x =  (index_pos[0] - 0.5) * 2.5
        y = -(index_pos[1] - 0.5) * 2.5
        z =  -index_pos[2] * 5.0
        return np.array([x, y, z])

    def upload_frame_to_texture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 0)
        h, w, _   = frame_rgb.shape
        glBindTexture(GL_TEXTURE_2D, self.cam_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

    def draw_camera_background(self):
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.cam_texture)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix(); glLoadIdentity(); glOrtho(0,1,0,1,-1,1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix(); glLoadIdentity()
        glColor3f(1,1,1)
        glBegin(GL_QUADS)
        glTexCoord2f(0,0); glVertex2f(0,0)
        glTexCoord2f(1,0); glVertex2f(1,0)
        glTexCoord2f(1,1); glVertex2f(1,1)
        glTexCoord2f(0,1); glVertex2f(0,1)
        glEnd()
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW);  glPopMatrix()
        glDisable(GL_TEXTURE_2D)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_hud_on_frame(self, frame, gesture, fingers, hand_labels=None):
        h, w  = frame.shape[:2]
        color = GESTURE_COLORS.get(gesture, (200,200,200))
        icon  = GESTURE_ICONS.get(gesture, "")
        hand_labels = hand_labels or []

        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,52), (15,15,25), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, f"MODE: {gesture}", (12,33),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"({icon})", (210,33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{self.fps:.0f} fps", (w-90,33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,160,160), 1, cv2.LINE_AA)
        if hand_labels:
            cv2.putText(frame, f"hands: {', '.join(hand_labels)}", (12,72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,200), 1, cv2.LINE_AA)

        for i, (lbl, state) in enumerate(zip(["T","I","M","R","P"], fingers)):
            cx = 560 + i * 24
            cv2.circle(frame, (cx,28), 9, color if state else (45,45,45), -1)
            cv2.putText(frame, lbl, (cx-4,33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (10,10,10), 1)

        if self.cube.selected_face is not None:
            face_names  = ["back","front","bottom","top","left","right"]
            face_label  = face_names[self.cube.selected_face]
            status_text = "EXTRUDING" if self.extruding else "selected"
            status_col  = (60,200,255) if self.extruding else (200,200,80)
            cv2.putText(frame, f"face: {face_label}  [{status_text}]",
                        (12, h-52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, status_col, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "no face selected", (12, h-52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1, cv2.LINE_AA)

        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0,h-34), (w,h), (15,15,25), -1)
        cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)
        pos = self.cube.position
        rot = self.cube.rotation
        cv2.putText(frame,
            f"pos({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})  "
            f"rot({rot[0]%360:.0f},{rot[1]%360:.0f},{rot[2]%360:.0f})  "
            f"scale {self.cube.scale:.2f}x",
            (10,h-11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,180), 1, cv2.LINE_AA)

        for i, hint in enumerate(["R: reset", "C: clear extrusions", "Q: quit"]):
            cv2.putText(frame, hint, (w-140,80+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110,110,130), 1, cv2.LINE_AA)
        return frame

    def ema(self, prev, curr):
        if prev is None:
            return np.array(curr)
        return self.alpha * np.array(curr) + (1 - self.alpha) * prev

    def reset_single_hand_motion(self):
        self.smooth_index = None
        self.smooth_wrist = None
        self.prev_smooth_index = None
        self.prev_smooth_wrist = None
        self.prev_pinch = None

    def reset_two_hand_motion(self):
        self.prev_two_center = None
        self.prev_two_distance = None
        self.prev_two_angle = None
        self.prev_two_depth = None
        self.prev_two_scale_distance = None

    def reset_extrude_motion(self):
        self.extruding = False
        self.extrude_face_idx = None
        self.prev_extrude_control = None

    def hand_label(self, handedness, fallback_idx):
        classifications = getattr(handedness, "classification", [])
        if classifications:
            return classifications[0].label
        return f"Hand {fallback_idx + 1}"

    def angle_delta(self, curr, prev):
        return (curr - prev + np.pi) % (2 * np.pi) - np.pi

    def extrude_control_value(self, hand_pos, face_idx):
        normal = self.cube.get_face_world_normal(face_idx)
        world_pos = self.finger_to_world(hand_pos)
        screen_axis = normal[:2]
        axis_len = np.linalg.norm(screen_axis)

        if axis_len > 0.2:
            return float(np.dot(world_pos[:2], screen_axis / axis_len))

        depth_facing_sign = 1.0 if normal[2] >= 0 else -1.0
        return float(world_pos[1] * depth_facing_sign)

    def apply_two_hand_gesture(self, hands):
        hand_a, hand_b = hands[:2]
        index_a = np.array(hand_a["index_pos"])
        index_b = np.array(hand_b["index_pos"])
        center = (index_a + index_b) * 0.5
        span = index_b - index_a
        distance = float(np.linalg.norm(span[:2]))
        angle = float(np.arctan2(span[1], span[0]))
        depth = float(center[2])

        gestures = {hand_a["gesture"], hand_b["gesture"]}
        both_grab = gestures == {"GRAB"}
        both_rotate = gestures == {"ROTATE"}
        scale_ready = all(h["fingers"][1] for h in (hand_a, hand_b))

        if both_grab:
            gesture = "TWO_MOVE"
        elif both_rotate:
            gesture = "TWO_ROTATE"
        elif scale_ready:
            gesture = "TWO_SCALE"
        else:
            gesture = "TWO_IDLE"

        if self.prev_two_center is None:
            self.prev_two_center = center.copy()
            self.prev_two_distance = distance
            self.prev_two_angle = angle
            self.prev_two_depth = depth
            return gesture

        dc = center - self.prev_two_center
        da = self.angle_delta(angle, self.prev_two_angle)
        dz = depth - self.prev_two_depth

        if gesture == "TWO_MOVE":
            self.extruding = False
            self.extrude_face_idx = None
            self.prev_two_scale_distance = None
            self.prev_extrude_control = None
            self.cube.position[0] += dc[0] * 4.0
            self.cube.position[1] += -dc[1] * 4.0
            self.cube.position[2] += dz * 12.0

        elif gesture == "TWO_SCALE":
            self.extruding = False
            self.prev_extrude_control = None
            if self.prev_two_scale_distance is None:
                self.prev_two_scale_distance = distance
            else:
                smoothed_distance = (
                    self.prev_two_scale_distance * 0.72 + distance * 0.28
                )
                delta = smoothed_distance - self.prev_two_scale_distance
                if abs(delta) > 0.012:
                    scale_step = float(np.clip(delta * 1.4, -0.015, 0.015))
                    self.cube.scale = float(
                        np.clip(self.cube.scale * (1.0 + scale_step), 0.15, 4.0)
                    )
                self.prev_two_scale_distance = smoothed_distance

        elif gesture == "TWO_ROTATE":
            self.extruding = False
            self.prev_extrude_control = None
            self.prev_two_scale_distance = None
            self.cube.rotation[1] += dc[0] * 300
            self.cube.rotation[0] += dc[1] * 300
            self.cube.rotation[2] += np.degrees(da)

        else:
            self.extruding = False
            self.prev_extrude_control = None
            self.prev_two_scale_distance = None

        self.prev_two_center = center.copy()
        self.prev_two_distance = distance
        self.prev_two_angle = angle
        self.prev_two_depth = depth
        return gesture

    def apply_gesture(self, data):
        gesture = data["gesture"]
        self.smooth_index = self.ema(self.smooth_index, data["index_pos"])
        self.smooth_wrist = self.ema(self.smooth_wrist, data["wrist_pos"])

        if self.prev_smooth_index is None:
            self.prev_smooth_index = self.smooth_index.copy()
            self.prev_smooth_wrist = self.smooth_wrist.copy()
            self.prev_pinch        = data["pinch_dist"]
            return

        di = self.smooth_index - self.prev_smooth_index
        dw = self.smooth_wrist  - self.prev_smooth_wrist

        if gesture == "POINT":
            self.reset_extrude_motion()
            finger_world = self.finger_to_world(self.smooth_index)
            selected_face = self.cube.select_face_at_pointer(finger_world[:2])
            if selected_face is not None:
                self.cube.selected_face = selected_face

        elif gesture == "DESELECT":
            self.cube.selected_face = None
            self.reset_extrude_motion()

        elif gesture == "GRAB":
            self.reset_extrude_motion()
            self.cube.position[0] +=  di[0] * 4.0
            self.cube.position[1] += -di[1] * 4.0
            self.cube.position[2] +=  di[2] * 12.0

        elif gesture == "EXTRUDE":
            if self.cube.selected_face is not None:
                self.extruding        = True
                self.extrude_face_idx = self.cube.selected_face
                control = self.extrude_control_value(
                    self.smooth_wrist, self.extrude_face_idx
                )
                if self.prev_extrude_control is None:
                    self.prev_extrude_control = control
                else:
                    delta = control - self.prev_extrude_control
                    if abs(delta) > 0.006:
                        delta = float(np.clip(delta * 0.9, -0.025, 0.025))
                        self.cube.extrude_face(self.extrude_face_idx, delta)
                    self.prev_extrude_control = control
            else:
                self.reset_extrude_motion()

        elif gesture == "ROTATE":
            self.reset_extrude_motion()
            self.cube.rotation[1] += dw[0] * 300
            self.cube.rotation[0] += dw[1] * 300

        elif gesture == "SCALE":
            self.reset_extrude_motion()

        else:
            self.reset_extrude_motion()

        self.prev_smooth_index = self.smooth_index.copy()
        self.prev_smooth_wrist = self.smooth_wrist.copy()
        self.prev_pinch        = data["pinch_dist"]

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.cube.position       = np.zeros(3)
                        self.cube.rotation       = np.zeros(3)
                        self.cube.scale          = 1.0
                        self.cube.vertex_offsets = np.zeros_like(self.cube.base_vertices)
                        self.cube.selected_face  = None
                        self.reset_extrude_motion()
                        self.reset_single_hand_motion()
                        self.reset_two_hand_motion()
                    if event.key == pygame.K_c:
                        self.cube.vertex_offsets = np.zeros_like(self.cube.base_vertices)
                        self.reset_extrude_motion()
                    if event.key == pygame.K_q:
                        running = False

            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb)

            gesture = "IDLE"
            fingers = [0,0,0,0,0]

            if results.multi_hand_landmarks:
                hands = []
                handedness_list = results.multi_handedness or []

                for idx, lms in enumerate(results.multi_hand_landmarks[:2]):
                    handedness = handedness_list[idx] if idx < len(handedness_list) else None
                    label = self.hand_label(handedness, idx)
                    engine = self.hand_engines.setdefault(label, GestureEngine(smooth_frames=5))
                    data = engine.get_landmarks(lms, label)
                    data["label"] = label
                    hands.append(data)

                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, lms, mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style(),
                    )

                hands.sort(key=lambda h: h["index_pos"][0])
                hand_labels = [h["label"] for h in hands]
                fingers = hands[0]["fingers"]

                if len(hands) >= 2:
                    self.reset_single_hand_motion()
                    gesture = self.apply_two_hand_gesture(hands)
                else:
                    self.reset_two_hand_motion()
                    gesture = hands[0]["gesture"]
                    self.apply_gesture(hands[0])
            else:
                self.reset_single_hand_motion()
                self.reset_two_hand_motion()
                hand_labels = []

            frame = self.draw_hud_on_frame(frame, gesture, fingers, hand_labels)

            cv2.imshow(self.hand_window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

            if self.show_camera_background:
                self.upload_frame_to_texture(frame)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            if self.show_camera_background:
                self.draw_camera_background()
            self.renderer.draw_scene()

            pygame.display.flip()
            self.clock.tick(60)
            self.fps = self.clock.get_fps()

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    AirCanvas3D().run()
