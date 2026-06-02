from ultralytics import YOLO
import cv2

# Load plate detection model
model = YOLO("models/yolov8n.pt")

# Open video
cap = cv2.VideoCapture("videos/traffic.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (960, 540))

    # Run plate detection
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("Plate Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()