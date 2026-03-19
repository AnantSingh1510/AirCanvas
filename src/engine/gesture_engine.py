import numpy as np

class GestureEngine:
    def __init__(self):
        self.tip_ids = [4, 8, 12, 16, 20]

    def fingers_up(self, hand_lms):
        tips = self.tip_ids
        fingers = []
        fingers.append(1 if hand_lms.landmark[tips[0]].x < hand_lms.landmark[tips[0]-1].x else 0)
        for i in range(1, 5):
            fingers.append(1 if hand_lms.landmark[tips[i]].y < hand_lms.landmark[tips[i]-2].y else 0)
        return fingers

    def get_landmarks(self, hand_lms):
        index = hand_lms.landmark[8]
        thumb  = hand_lms.landmark[4]
        middle = hand_lms.landmark[12]
        wrist  = hand_lms.landmark[0]

        pinch_dist = np.linalg.norm([index.x - thumb.x, index.y - thumb.y])
        fingers = self.fingers_up(hand_lms)

        if fingers == [0, 0, 0, 0, 0]:
            gesture = "GRAB"
        elif fingers == [0, 1, 0, 0, 0]:
            gesture = "DRAW"
        elif fingers == [0, 1, 1, 0, 0]:
            gesture = "ROTATE" 
        elif fingers == [1, 1, 0, 0, 0]:
            gesture = "SCALE"
        elif fingers == [1, 1, 1, 1, 1]:
            gesture = "IDLE"
        else:
            gesture = "IDLE"

        return {
            "index_pos":  (index.x,  index.y,  index.z),
            "middle_pos": (middle.x, middle.y, middle.z),
            "wrist_pos":  (wrist.x,  wrist.y,  wrist.z),
            "pinch_dist": pinch_dist,
            "gesture":    gesture,
            "is_drawing":  gesture == "DRAW",
            "is_rotating": gesture == "ROTATE",
        }