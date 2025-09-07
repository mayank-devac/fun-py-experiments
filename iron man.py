import cv2
import mediapipe as mp
import numpy as np
import math
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)

# Function to draw the holographic UI
def draw_holographic_ui(image, hand_landmarks):
    # Get image dimensions
    h, w, c = image.shape

    # --- 1. Central Palm Element ---
    # Use wrist and middle finger MCP as anchor points
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    # Calculate center of the palm for the UI
    palm_center_x = int((wrist.x + middle_mcp.x) / 2 * w)
    palm_center_y = int((wrist.y + middle_mcp.y) / 2 * h)

    # Draw concentric glowing circles
    cv2.circle(image, (palm_center_x, palm_center_y), 40, (0, 255, 255), 2) # Yellow
    cv2.circle(image, (palm_center_x, palm_center_y), 35, (255, 0, 150), 2) # Pink
    cv2.circle(image, (palm_center_x, palm_center_y), 20, (0, 255, 255), -1) # Center fill

    # --- 2. Finger Tracking Lines and Dots ---
    for i in [4, 8, 12, 16, 20]: # Finger tips
        tip = hand_landmarks.landmark[i]
        tip_x, tip_y = int(tip.x * w), int(tip.y * h)
        cv2.line(image, (palm_center_x, palm_center_y), (tip_x, tip_y), (255, 255, 0), 1) # Cyan lines
        cv2.circle(image, (tip_x, tip_y), 8, (0, 255, 0), -1) # Green dots

    # --- 3. Dynamic Arc Gauge ---
    # Calculate distance between thumb and index finger tip
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
    
    # Map distance to an angle for the arc
    # Max distance is around 0.3-0.4 in normalized coordinates
    arc_angle = int(np.interp(distance, [0.05, 0.3], [0, 270]))
    
    # Draw the arc gauge around the hand
    cv2.ellipse(image, (palm_center_x, palm_center_y), (100, 100), 0, -90, arc_angle - 90, (255, 0, 0), 4) # Blue arc
    
    # --- 4. Floating Data Points ---
    # Display a random number near the index finger
    data_point_x = int(hand_landmarks.landmark[8].x * w) + 20
    data_point_y = int(hand_landmarks.landmark[8].y * h)
    random_data = random.randint(20, 100)
    cv2.putText(image, str(random_data), (data_point_x, data_point_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(rgb_image)
    
    # Create a black background to draw the UI on for a "hologram" effect
    overlay = np.zeros(image.shape, dtype=np.uint8)

    # Draw hand landmarks and connections
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the custom holographic UI on the black overlay
            draw_holographic_ui(overlay, hand_landmarks)
            
            # Optional: Draw the default MediaPipe hand skeleton for reference
            # mp_drawing.draw_landmarks(
            #     image,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS)
    
    # Add the overlay to the original image
    final_image = cv2.addWeighted(image, 1.0, overlay, 0.8, 0)

    cv2.imshow('Holographic Hand UI', final_image)

    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()