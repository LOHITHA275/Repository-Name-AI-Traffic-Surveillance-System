from ultralytics import YOLO
import supervision as sv
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("videos/traffic.mp4")

# Initialize ByteTrack
tracker = sv.ByteTrack()

# Box annotator
box_annotator = sv.BoxAnnotator()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO detection
    results = model(frame)[0]

    # Convert results to supervision detections
    detections = sv.Detections.from_ultralytics(results)

    # Update tracker
    detections = tracker.update_with_detections(detections)

    # Labels with IDs
    labels = [
        f"ID {tracker_id}"
        for tracker_id in detections.tracker_id
    ]

    # Annotate frame
    annotated_frame = box_annotator.annotate(
        scene=frame,
        detections=detections
    )

    # Draw labels
    for i, box in enumerate(detections.xyxy):
        x1, y1, x2, y2 = map(int, box)

        cv2.putText(
            annotated_frame,
            labels[i],
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Show frame
    cv2.imshow("Vehicle Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()