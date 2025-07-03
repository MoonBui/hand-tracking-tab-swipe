from flask import Flask, render_template, Response
import cv2
import time
import HandTracking as ht
import SwipeTracking as st
import numpy as np
import math
import KeyRegister as kr

# Initialize camera and detector
cap = cv2.VideoCapture(0)
detector = ht.HandDetector(detectionCon=0.75)

# Performance settings
DETECTION_SKIP_FRAMES = 3  # Only detect every 3rd frame
fingers_to_track = [8, 12]

# Cache for storing detection results
last_detection_results = {
    'lmList': [],
    'bbox': None,
    'active_fingers': [],
    'swipes': []
}

def gen_frames():
    pTime = 0
    frame_count = 0
    
    while True:
        success, img = cap.read()
        frame_count += 1

        if not success:
            break
        
        # PERFORMANCE: Only do heavy processing every N frames
        if frame_count % DETECTION_SKIP_FRAMES == 0:
            # Full hand detection and processing
            img = detector.findHands(img)
            lmList, bbox = detector.findPosition(img)
            
            if len(lmList) != 0:
                # Update swipe tracking
                detector.update_swipe_tracking(fingers_to_track)
                
                # Check for swipes and process immediately (no threading)
                swipes = detector.check_swipes(fingers_to_track)
                
                # Process swipes directly - keep it simple
                for swipe in swipes:
                    finger_name = {8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky", 4: "Thumb"}
                    name = finger_name.get(swipe['finger_id'], f"Finger {swipe['finger_id']}")
                    kr.register_key(swipe['direction'])
                    
                    print(f"{name} finger swiped {swipe['direction']} - "
                          f"Distance: {swipe['distance']:.1f}px, "
                          f"Speed: {swipe['velocity']:.1f}px/s")
                
                # Cache results for non-detection frames
                last_detection_results['lmList'] = lmList
                last_detection_results['bbox'] = bbox
                last_detection_results['active_fingers'] = detector.get_swipe_tracker().get_all_active_fingers()
                last_detection_results['swipes'] = swipes
        
        # Always draw visual elements (using cached data on non-detection frames)
        if len(last_detection_results['lmList']) != 0:
            # Draw swipe trails
            detector.draw_swipe_trails(img, fingers_to_track)
            
            # Show active finger count
            cv2.putText(img, f"Active: {len(last_detection_results['active_fingers'])}", 
                       (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
            
            # Show recent swipes
            for i, swipe in enumerate(last_detection_results['swipes'][-3:]):  # Show last 3 swipes
                finger_name = {8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky", 4: "Thumb"}
                name = finger_name.get(swipe['finger_id'], f"Finger {swipe['finger_id']}")
                cv2.putText(img, f"{name}: {swipe['direction']}", 
                           (10, 120 + i * 30), 
                           cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Simple FPS calculation
        cTime = time.time()
        fps = 1/(cTime-pTime) if pTime != 0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        # Lightweight frame encoding
        ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)