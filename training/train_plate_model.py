from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="../datasets/license_plate/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    name="license_plate_detector_fast"
)