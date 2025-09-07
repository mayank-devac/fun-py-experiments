import cv2
from cvzone.HandTrackingModule import HandDetector

# Start webcam capture
cap = cv2.VideoCapture(0)

# Initialize Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()

    # Flip the image for a natural view
    img = cv2.flip(img, 1)

    # Find hands and draw landmarks
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if a hand is detected
    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]

        # Use the position of the landmarks for your UI
        # lmList1[8] is the index finger tip, for example.
        # You can build the UI from here. For simplicity, we just draw.

        # Example: Draw a circle at the index finger tip
        cv2.circle(img, (lmList1[8][0], lmList1[8][1]), 15, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()