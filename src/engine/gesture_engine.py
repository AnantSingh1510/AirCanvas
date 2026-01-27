import numpy as np

class GestureEngine:
    def __init__(self):
        self.tip_ids = [4, 8, 12, 16, 20]

    def get_landmarks(self, hand_lms):
        index = hand_lms.landmark[8]
        middle = hand_lms.landmark[12]
        
        thumb = hand_lms.landmark[4]
        dist = np.linalg.norm(np.array([index.x - thumb.x, index.y - thumb.y]))
        
        return {
            "index_pos": (index.x, index.y, index.z),
            "is_drawing": dist > 0.1,
            "is_rotating": middle.y < hand_lms.landmark[10].y
        }