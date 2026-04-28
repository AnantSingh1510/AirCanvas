import cv2
import numpy as np

class CanvasEngine:
    def __init__(self, width, height):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        self.color_idx = 0
        self.prev_point = None

    def draw(self, point, state, thickness=5):
        if state == "DRAW":
            if self.prev_point:
                cv2.line(self.canvas, self.prev_point, point, self.colors[self.color_idx], thickness)
            self.prev_point = point
        elif state == "ERASE":
            cv2.circle(self.canvas, point, 50, (0, 0, 0), -1)
            self.prev_point = None
        else:
            self.prev_point = None

    def clear(self):
        self.canvas = np.zeros_like(self.canvas)

    def get_canvas(self):
        return self.canvas