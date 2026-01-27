import cv2
import mediapipe as mp
import numpy as np

class AirCanvas:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.canvas = None

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)

            # Processing logic - will add later - Anant Singh
            
            cv2.imshow("Air Canvas Pro", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AirCanvas()
    app.run()