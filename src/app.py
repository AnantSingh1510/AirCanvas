import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from engine.gesture_engine import GestureEngine
from engine.gl_renderer import GLRenderer
import websockets
import asyncio
import json

class AirCanvas3D:
    def __init__(self):
        pygame.init()
        self.display = (800, 600)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.7)
        self.engine = GestureEngine()
        self.renderer = GLRenderer()
        self.renderer.init_gl(*self.display)
        
        self.rotation_y = 0

    async def stream_to_web(self, x, y, z, state):
        uri = "ws://localhost:8000/ws/canvas"
        async with websockets.connect(uri) as websocket:
            data = {
                "x": x,
                "y": y,
                "z": z,
                "state": state
            }
            await websocket.send(json.dumps(data))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # Process Hand Tracking
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                data = self.engine.get_landmarks(results.multi_hand_landmarks[0])
                
                if data["is_rotating"]:
                    self.rotation_y += 2
                
                if data["is_drawing"]:
                    self.renderer.current_stroke.append(data["index_pos"])
                else:
                    if self.renderer.current_stroke:
                        self.renderer.lines.append(self.renderer.current_stroke)
                        self.renderer.current_stroke = []

            self.renderer.draw_scene(self.rotation_y)
            pygame.display.flip()
            pygame.time.wait(10)

        self.cap.release()
        pygame.quit()

if __name__ == "__main__":
    AirCanvas3D().run()