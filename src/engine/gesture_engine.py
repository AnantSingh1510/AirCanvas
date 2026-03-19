import numpy as np
from collections import deque

class GestureEngine:
    def __init__(self, smooth_frames=5):
        self.tip_ids = [4, 8, 12, 16, 20]
        self.landmark_history = {i: deque(maxlen=smooth_frames) for i in range(21)}
        self.gesture_buffer   = deque(maxlen=6)
        self.confirmed_gesture = "IDLE"

    def smooth_landmarks(self, hand_lms):
        for i in range(21):
            lm = hand_lms.landmark[i]
            self.landmark_history[i].append((lm.x, lm.y, lm.z))
        return {i: np.mean(self.landmark_history[i], axis=0) for i in range(21)}

    def fingers_up(self, s):
        fingers = []
        fingers.append(1 if s[4][0] < s[3][0] else 0)   # thumb
        for tip, base in [(8,6),(12,10),(16,14),(20,18)]:
            fingers.append(1 if s[tip][1] < s[base][1] else 0)
        return fingers

    def classify_gesture(self, fingers, pinch_dist):
        f = fingers
        if   f == [0, 0, 0, 0, 0]:                        return "GRAB"
        elif f == [0, 1, 0, 0, 0]:                        return "POINT"
        elif f == [0, 1, 1, 0, 0]:                        return "ROTATE"
        elif f == [0, 1, 1, 1, 0]:                        return "DESELECT"
        elif f == [1, 1, 0, 0, 0] and pinch_dist < 0.12:  return "SCALE"
        elif f == [1, 1, 1, 1, 1]:                        return "IDLE"
        else:                                              return "IDLE"

    def stable_gesture(self, raw):
        self.gesture_buffer.append(raw)
        counts = {}
        for g in self.gesture_buffer:
            counts[g] = counts.get(g, 0) + 1
        best = max(counts, key=counts.get)
        if counts[best] >= 4:
            self.confirmed_gesture = best
        return self.confirmed_gesture

    def get_landmarks(self, hand_lms):
        s          = self.smooth_landmarks(hand_lms)
        index      = s[8]
        thumb      = s[4]
        middle     = s[12]
        wrist      = s[0]
        pinch_dist = np.linalg.norm(index[:2] - thumb[:2])
        fingers    = self.fingers_up(s)
        gesture    = self.stable_gesture(self.classify_gesture(fingers, pinch_dist))

        return {
            "index_pos":  tuple(index),
            "middle_pos": tuple(middle),
            "wrist_pos":  tuple(wrist),
            "thumb_pos":  tuple(thumb),
            "pinch_dist": float(pinch_dist),
            "gesture":    gesture,
            "fingers":    fingers,
            "is_drawing":  False,
            "is_rotating": gesture == "ROTATE",
        }