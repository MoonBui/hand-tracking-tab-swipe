import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
from SwipeTracking import SwipeTracker

# base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
# options = vision.GestureRecognizerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.LIVE_STREAM,
#     result_callback=save_result)


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=1,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.track_toggle = False
        self.last_gesture_time = 0
        self.gesture_cooldown = 2

        self.swipe_tracker = SwipeTracker(
            max_points=10,
            min_distance=40,
            min_velocity=50
        )

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            
            # Only calculate bbox if we have landmarks
            if xList and yList:
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax

                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                    (0, 255, 0), 2)

        return self.lmList, bbox
        
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
    
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # totalFingers = fingers.count(1)

        return fingers
    
    def update_swipe_tracking(self, fingers_to_track=None):
        if len(self.lmList) == 0:
            return
        
        if fingers_to_track is None:
            fingers_to_track = self.tipIds
        
        for finger_id in fingers_to_track:
            if finger_id < len(self.lmList):
                finger_pos = (self.lmList[finger_id][1], self.lmList[finger_id][2])
                self.swipe_tracker.add_point(finger_id, finger_pos)
        
        self.swipe_tracker.cleanup_inactive_trails()


    def check_swipes(self, fingers_to_check=None):
        """
        Check for swipe gestures on specified fingers.
        Returns list of detected swipes.
        """
        if fingers_to_check is None:
            fingers_to_check = self.tipIds
        
        detected_swipes = []
        for finger_id in fingers_to_check:
            swipe = self.swipe_tracker.detect_swipe(finger_id)
            if swipe:
                detected_swipes.append(swipe)
                # Optionally clear the trail after detection
                self.swipe_tracker.clear_trail(finger_id)
        
        return detected_swipes
    
    def draw_swipe_trails(self, img, fingers_to_draw=None):
        """Draw swipe trails for specified fingers."""
        if fingers_to_draw is None:
            self.swipe_tracker.draw_all_trails(img)
        else:
            for finger_id in fingers_to_draw:
                self.swipe_tracker.draw_trail(img, finger_id)
    
    def get_swipe_tracker(self):
        """Get direct access to the swipe tracker for advanced usage."""
        return self.swipe_tracker
    

    def detect_victory_gesture(self):
        if len(self.lmList) == 0:
            return False
        fingers = self.fingersUp()

        if fingers != [0, 1, 1, 0, 0]:
            return False
        
        index_tip = self.lmList[8]  # index finger tip
        middle_tip = self.lmList[12]  # middle finger tip
        
        # Calculate distance between index and middle finger tips
        distance = math.hypot(index_tip[1] - middle_tip[1], index_tip[2] - middle_tip[2])
        
        # They should be reasonably spread apart for a proper victory sign
        if distance > 30: 
            print("Victory gesture detected")
        return distance > 30

    def update_track_toggle(self):
        current_time = time.time()
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return self.track_toggle
        if self.detect_victory_gesture():
            self.track_toggle = not self.track_toggle
            self.last_gesture_time = current_time
        return self.track_toggle
    
    def draw_tracking_status(self, img):
        status_text = f"Tracking: {'ON' if self.track_toggle else 'OFF'}"
        color = (0, 255, 0) if self.track_toggle else (0, 0, 255)
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        instructions_text = "Victory sign to toggle swipe tracking"
        cv2.putText(img, instructions_text, (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)


# Templating stuff
# 
# 
#  def main():
#     cap = cv2.VideoCapture(0)
#     pTime = 0
#     cTime = 0
#     detector = HandDetector()

#     while True:
#         success, img = cap.read()
#         img = detector.findHands(img)
#         lmList, bbox = detector.findPosition(img)
#         if len(lmList) != 0:
#             print(lmList[4])

#         cTime = time.time()
#         fps = 1/(cTime-pTime)
#         pTime = cTime
#         cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

#         cv2.imshow("Image", img)
#         cv2.waitKey(1)



# if __name__ == "__main__":
#     main()