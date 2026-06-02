from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video file
video_path = "videos/traffic.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened
if not cap.isOpened():
    print("Error opening video")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended")
        break
    frame = cv2.resize(frame, (800, 450))

    # Run YOLO detection
    results = model(frame)

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Vehicle Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()