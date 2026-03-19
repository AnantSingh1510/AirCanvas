import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from engine.gesture_engine import GestureEngine
from engine.gl_renderer import GLRenderer
from engine.cube import Cube
import numpy as np

class AirCanvas3D:
    def __init__(self):
        pygame.init()
        self.display = (800, 600)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)

        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.7)
        self.engine  = GestureEngine()
        self.renderer = GLRenderer()
        self.renderer.init_gl(*self.display)

        self.cube = Cube(size=0.5)
        self.renderer.cube = self.cube

        self.rotation_y   = 0
        self.prev_data     = None

    def apply_gesture(self, data):
        gesture = data["gesture"]
        ix, iy, iz = data["index_pos"]

        if self.prev_data is None:
            self.prev_data = data
            return

        px, py, pz = self.prev_data["index_pos"]
        dx = (ix - px) * 3
        dy = (iy - py) * 3
        dz = (iz - pz) * 10

        if gesture == "GRAB":
            self.cube.position[0] += dx
            self.cube.position[1] -= dy
            self.cube.position[2] += dz

        elif gesture == "ROTATE":
            wx, wy, _ = data["wrist_pos"]
            pwx, pwy, _ = self.prev_data["wrist_pos"]
            self.cube.rotation[1] += (wx - pwx) * 200
            self.cube.rotation[0] += (wy - pwy) * 200

        elif gesture == "SCALE":
            scale_delta = (data["pinch_dist"] - self.prev_data["pinch_dist"]) * 5
            self.cube.scale = max(0.1, min(5.0, self.cube.scale + scale_delta))

        elif gesture == "DRAW":
            self.renderer.current_stroke.append(data["index_pos"])

        else:
            if self.renderer.current_stroke:
                self.renderer.lines.append(self.renderer.current_stroke)
                self.renderer.current_stroke = []

        self.prev_data = data

    def draw_hud(self, frame, gesture):
        """Overlay gesture name and controls onto the webcam preview."""
        colors = {
            "DRAW": (0, 255, 0), "ROTATE": (255, 165, 0),
            "GRAB": (0, 120, 255), "SCALE": (255, 0, 200), "IDLE": (180, 180, 180)
        }
        color = colors.get(gesture, (255, 255, 255))
        cv2.putText(frame, f"Mode: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, "INDEX=Draw  2FIN=Rotate  FIST=Move  THUMB+IDX=Scale",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1)
        return frame

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.cube.position = np.array([0.0, 0.0, 0.0])
                        self.cube.rotation = np.array([0.0, 0.0, 0.0])
                        self.cube.scale    = 1.0
                    if event.key == pygame.K_c:
                        self.renderer.lines = []
                        self.renderer.current_stroke = []

            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = self.mp_hands.process(rgb_frame)

            gesture = "IDLE"
            if results.multi_hand_landmarks:
                data    = self.engine.get_landmarks(results.multi_hand_landmarks[0])
                gesture = data["gesture"]
                self.apply_gesture(data)

            frame = self.draw_hud(frame, gesture)
            cv2.imshow("AirCanvas — Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

            self.renderer.draw_scene(self.rotation_y)
            pygame.display.flip()
            pygame.time.wait(10)

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    AirCanvas3D().run()