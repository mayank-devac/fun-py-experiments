import cv2
import mediapipe as mp
import numpy as np
import random  # For wiggly lines
import math    # For angles
import threading    


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Colors (BGR): teal, orange, red
TEAL = (255, 255, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
GLOW_ALPHA = 0.6  # For blended glow

# Helper: Draw glowing line (multi-layer for glow effect)
def draw_glow_line(img, pt1, pt2, color, thickness=2, glow_layers=3):
    for i in range(glow_layers):
        alpha = GLOW_ALPHA / (i + 1)
        overlay = img.copy()
        cv2.line(overlay, pt1, pt2, color, thickness + i * 2)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# Helper: Draw wiggly line (add noise for organic feel)
def draw_wiggly_line(img, start, end, color, segments=5, noise=5):
    points = [start]
    for i in range(1, segments):
        t = i / segments
        x = int(start[0] * (1 - t) + end[0] * t + random.uniform(-noise, noise))
        y = int(start[1] * (1 - t) + end[1] * t + random.uniform(-noise, noise))
        points.append((x, y))
    points.append(end)
    for i in range(len(points) - 1):
        draw_glow_line(img, points[i], points[i+1], color)

# Helper: Draw glowing circle
def draw_glow_circle(img, center, radius, color, thickness=2, glow_layers=3):
    for i in range(glow_layers):
        alpha = GLOW_ALPHA / (i + 1)
        overlay = img.copy()
        cv2.circle(overlay, center, radius + i * 2, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# Capture video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hand
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lms = hand_lms.landmark
            
            # Calculate palm center (avg of wrist + MCP joints)
            palm_x = sum([lms[i].x for i in [0, 5, 9, 13, 17]]) / 5
            palm_y = sum([lms[i].y for i in [0, 5, 9, 13, 17]]) / 5
            palm = (int(palm_x * w), int(palm_y * h))
            
            # Draw palm UI: Central circle with icon (simple ellipse for 'brain')
            draw_glow_circle(frame, palm, 40, TEAL)
            draw_glow_circle(frame, palm, 30, ORANGE, thickness=1)
            cv2.ellipse(frame, (palm[0], palm[1] - 5), (15, 10), 0, 0, 360, ORANGE, -1)  # Icon placeholder
            
            # Concentric rings
            draw_glow_circle(frame, palm, 50, TEAL, thickness=1)
            draw_glow_circle(frame, palm, 60, RED, thickness=1)
            
            # Custom finger connections (wiggly orange lines, green circles)
            connections = mp_hands.HAND_CONNECTIONS
            for conn in connections:
                pt1 = (int(lms[conn[0]].x * w), int(lms[conn[0]].y * h))
                pt2 = (int(lms[conn[1]].x * w), int(lms[conn[1]].y * h))
                draw_wiggly_line(frame, pt1, pt2, ORANGE)
            
            # Green circles at joints
            for lm in lms:
                pt = (int(lm.x * w), int(lm.y * h))
                draw_glow_circle(frame, pt, 5, (0, 255, 0), thickness=-1)
            
            # Gesture: Pinch detection (thumb tip #4 + index tip #8)
            thumb = (int(lms[4].x * w), int(lms[4].y * h))
            index = (int(lms[8].x * w), int(lms[8].y * h))
            dist = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
            pinch_threshold = 30  # Pixels
            
            if dist < pinch_threshold:
                # Pinch active: Draw radial selector line + angle display
                pinch_pt = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)
                draw_glow_line(frame, palm, pinch_pt, RED, thickness=3)
                
                # Calculate angle (from palm to pinch, relative to horizontal)
                dx = pinch_pt[0] - palm[0]
                dy = pinch_pt[1] - palm[1]
                angle = int(math.degrees(math.atan2(dy, dx)))
                cv2.putText(frame, f"{angle}°", (pinch_pt[0] + 10, pinch_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
                
                # Radial bars (simulated UI feedback)
                for i in range(8):  # 8 segments
                    rad = math.radians(angle + i * 45)
                    end_x = palm[0] + int(70 * math.cos(rad))
                    end_y = palm[1] + int(70 * math.sin(rad))
                    draw_glow_line(frame, palm, (end_x, end_y), ORANGE, thickness=1)
            
            # Simple finger count display (example interactive element)
            open_fingers = sum(1 for i in [8,12,16,20] if lms[i].y < lms[i-2].y)  # Tips above PIP
            cv2.putText(frame, f"Fingers: {open_fingers + 1}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, TEAL, 2)  # +thumb
            
    # Show frame
    cv2.imshow("Holographic Hand UI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()