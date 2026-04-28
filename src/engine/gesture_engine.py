import numpy as np
from collections import deque

class GestureEngine:
    def __init__(self, smooth_frames=5):
        self.tip_ids = [4, 8, 12, 16, 20]
        self.landmark_history = {i: deque(maxlen=smooth_frames) for i in range(21)}
        self.gesture_buffer   = deque(maxlen=4)
        self.confirmed_gesture = "IDLE"

    def smooth_landmarks(self, hand_lms):
        for i in range(21):
            lm = hand_lms.landmark[i]
            self.landmark_history[i].append((lm.x, lm.y, lm.z))
        return {i: np.mean(self.landmark_history[i], axis=0) for i in range(21)}

    def joint_angle(self, a, b, c):
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom == 0:
            return 0.0
        cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def is_finger_extended(self, s, mcp, pip, tip):
        wrist = s[0]
        angle = self.joint_angle(s[mcp], s[pip], s[tip])
        tip_dist = np.linalg.norm(s[tip] - wrist)
        pip_dist = np.linalg.norm(s[pip] - wrist)
        mcp_dist = np.linalg.norm(s[mcp] - wrist)
        return (
            angle > 138 and tip_dist > pip_dist * 0.98
        ) or (
            angle > 125 and tip_dist > mcp_dist * 1.22
        )

    def is_thumb_extended(self, s):
        wrist = s[0]
        angle = self.joint_angle(s[2], s[3], s[4])
        tip_dist = np.linalg.norm(s[4] - wrist)
        ip_dist = np.linalg.norm(s[3] - wrist)
        return angle > 130 and tip_dist > ip_dist * 0.94

    def fingers_up(self, s, handedness_label=None):
        return [
            1 if self.is_thumb_extended(s) else 0,
            1 if self.is_finger_extended(s, 5, 6, 8) else 0,
            1 if self.is_finger_extended(s, 9, 10, 12) else 0,
            1 if self.is_finger_extended(s, 13, 14, 16) else 0,
            1 if self.is_finger_extended(s, 17, 18, 20) else 0,
        ]

    def classify_gesture(self, fingers, pinch_dist, s=None):
        f = fingers
        thumb, index, middle, ring, pinky = f
        folded_fingers = 4 - sum(f[1:])
        index_reach = 0.0
        middle_reach = 0.0
        if s is not None:
            wrist = s[0]
            index_reach = np.linalg.norm(s[8] - wrist)
            middle_reach = np.linalg.norm(s[12] - wrist)
        middle_is_deliberate = s is None or middle_reach > index_reach * 0.82

        if pinch_dist < 0.10 and thumb and index:
            return "SCALE"
        if thumb and pinky and not index and not middle:
            return "EXTRUDE"
        if index and middle and ring and not pinky:
            return "DESELECT"
        if index and middle and not ring and not pinky and middle_is_deliberate:
            return "ROTATE"
        if index and folded_fingers >= 2:
            return "POINT"
        if index and middle and not ring and not pinky:
            return "POINT"
        if sum(f[1:]) == 4:
            return "IDLE"
        if sum(f) <= 1 and not index:
            return "GRAB"
        return self.confirmed_gesture if self.confirmed_gesture != "IDLE" else "IDLE"

    def stable_gesture(self, raw):
        self.gesture_buffer.append(raw)
        counts = {}
        for g in self.gesture_buffer:
            counts[g] = counts.get(g, 0) + 1
        best = max(counts, key=counts.get)
        if counts[best] >= 2 or raw == self.confirmed_gesture:
            self.confirmed_gesture = best
        return self.confirmed_gesture

    def get_landmarks(self, hand_lms, handedness_label=None):
        s          = self.smooth_landmarks(hand_lms)
        index      = s[8]
        thumb      = s[4]
        middle     = s[12]
        wrist      = s[0]
        pinch_dist = np.linalg.norm(index[:2] - thumb[:2])
        fingers    = self.fingers_up(s, handedness_label)
        gesture    = self.stable_gesture(self.classify_gesture(fingers, pinch_dist, s))

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
