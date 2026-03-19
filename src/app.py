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
    "SCALE":    (220,  60, 220),
    "DESELECT": (255,  80,  80),
    "IDLE":     (160, 160, 160),
}
GESTURE_ICONS = {
    "POINT":    "index — select face",
    "ROTATE":   "2 fingers — rotate",
    "GRAB":     "fist — move / extrude",
    "SCALE":    "thumb+idx — scale",
    "DESELECT": "3 fingers — deselect",
    "IDLE":     "open palm",
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
            max_num_hands=1
        )
        self.engine   = GestureEngine(smooth_frames=5)
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

        self.extruding        = False
        self.extrude_face_idx = None

        self.clock = pygame.time.Clock()
        self.fps   = 0
        self.cam_texture = glGenTextures(1)

        cv2.namedWindow("AirCanvas — Hand View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AirCanvas — Hand View", 480, 360)

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
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_hud_on_frame(self, frame, gesture, fingers):
        h, w  = frame.shape[:2]
        color = GESTURE_COLORS.get(gesture, (200,200,200))
        icon  = GESTURE_ICONS.get(gesture, "")

        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,52), (15,15,25), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, f"MODE: {gesture}", (12,33),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"({icon})", (210,33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{self.fps:.0f} fps", (w-90,33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,160,160), 1, cv2.LINE_AA)

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
            finger_world = self.finger_to_world(data["index_pos"])
            self.cube.selected_face = self.cube.select_nearest_face(finger_world)
            self.extruding = False

        elif gesture == "DESELECT":
            self.cube.selected_face = None
            self.extruding          = False
            self.extrude_face_idx   = None

        elif gesture == "GRAB":
            if self.cube.selected_face is not None and not self.extruding:
                self.extruding        = True
                self.extrude_face_idx = self.cube.selected_face

            if self.extruding and self.extrude_face_idx is not None:
                normal   = self.cube.get_face_world_normal(self.extrude_face_idx)
                movement = np.array([di[0], -di[1], di[2] * 3])
                delta    = float(np.dot(movement, normal)) * 2.5
                self.cube.extrude_face(self.extrude_face_idx, delta)
            else:
                self.cube.position[0] +=  di[0] * 4.0
                self.cube.position[1] += -di[1] * 4.0
                self.cube.position[2] +=  di[2] * 12.0

        elif gesture == "ROTATE":
            self.extruding = False
            self.cube.rotation[1] += dw[0] * 300
            self.cube.rotation[0] += dw[1] * 300

        elif gesture == "SCALE":
            self.extruding = False
            dpinch = data["pinch_dist"] - (self.prev_pinch or data["pinch_dist"])
            self.cube.scale = float(np.clip(self.cube.scale + dpinch * 6, 0.15, 4.0))

        else:
            self.extruding = False

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
                        self.smooth_index        = None
                        self.smooth_wrist        = None
                        self.extruding           = False
                        self.extrude_face_idx    = None
                    if event.key == pygame.K_c:
                        self.cube.vertex_offsets = np.zeros_like(self.cube.base_vertices)
                        self.extruding           = False
                        self.extrude_face_idx    = None
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
                lms     = results.multi_hand_landmarks[0]
                data    = self.engine.get_landmarks(lms)
                gesture = data["gesture"]
                fingers = data["fingers"]
                self.apply_gesture(data)

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, lms, mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )

            frame = self.draw_hud_on_frame(frame, gesture, fingers)

            cv2.imshow("AirCanvas Hand View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

            self.upload_frame_to_texture(frame)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
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