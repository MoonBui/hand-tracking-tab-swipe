import cv2
import time
import HandTracking as ht
import SwipeTracking as st
import numpy as np
import math
import KeyRegister as kr

cap = cv2.VideoCapture(0)
detector = ht.HandDetector(detectionCon=0.75)
pTime = 0
cTime = 0
xp, yp = 0, 0
fingers_to_track = [8, 12]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    
    if len(lmList) != 0:
        # Update swipe tracking
        detector.update_swipe_tracking(fingers_to_track)
        
        # Check for swipes
        swipes = detector.check_swipes(fingers_to_track)
        
        # Handle detected swipes
        for swipe in swipes:
            finger_name = {8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky", 4: "Thumb"}
            name = finger_name.get(swipe['finger_id'], f"Finger {swipe['finger_id']}")
            kr.register_key(swipe['direction'])
            
            print(f"{name} finger swiped {swipe['direction']} - "
                    f"Distance: {swipe['distance']:.1f}px, "
                    f"Speed: {swipe['velocity']:.1f}px/s")
            
            # Display swipe on screen
            cv2.putText(img, f"{name}: {swipe['direction']}", 
                        (10, 120 + len([s for s in swipes if s == swipe]) * 30), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        # Draw swipe trails
        detector.draw_swipe_trails(img, fingers_to_track)
        
        # Show active finger count
        active_fingers = detector.get_swipe_tracker().get_all_active_fingers()
        cv2.putText(img, f"Active: {len(active_fingers)}", (10, 150), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

    # Display FPS
    cTime = time.time()
    fps = 1/(cTime-pTime) if pTime != 0 else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("Hand Tracking with Swipe Detection", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        








