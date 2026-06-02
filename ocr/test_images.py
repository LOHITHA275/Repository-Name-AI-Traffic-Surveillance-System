from ultralytics import YOLO
import easyocr
import cv2
import os
import re

# =========================
# LOAD MODEL
# =========================
model = YOLO("yolov8n.pt")

# =========================
# OCR
# =========================
reader = easyocr.Reader(['en'])

# =========================
# IMAGE FOLDER
# =========================
image_folder = "videos/plate_pic/images"

# =========================
# LOOP THROUGH IMAGES
# =========================
for image_name in os.listdir(image_folder):

    image_path = os.path.join(image_folder, image_name)

    print(f"\nProcessing: {image_name}")

    # Read image
    frame = cv2.imread(image_path)

    # Skip invalid images
    if frame is None:
        print("Could not read image")
        continue

    # Resize
    frame = cv2.resize(frame, (960, 540))

    # =========================
    # DETECTION
    # =========================
    results = model(frame)[0]

    print("Detections:", len(results.boxes))

    # =========================
    # LOOP DETECTIONS
    # =========================
    for box in results.boxes:

        # Coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Confidence
        confidence = float(box.conf[0])

        # Ignore weak detections
        if confidence < 0.3:
            continue

        # Crop detected vehicle
        vehicle_crop = frame[y1:y2, x1:x2]

        # Skip invalid crop
        if vehicle_crop.size == 0:
            continue

        # =========================
        # OCR
        # =========================
        try:
            ocr_results = reader.readtext(vehicle_crop)

        except Exception as e:
            print("OCR Error:", e)
            continue

        detected_text = ""

        # Extract OCR text
        for result in ocr_results:

            text = result[1]

            # Clean text
            text = re.sub(r'[^A-Z0-9]', '', text.upper())

            # Keep only realistic plate-like text
        if len(text) >= 6 and len(text) <= 12:
            detected_text += text + " "

        detected_text = detected_text.strip()
        # Skip nonsense OCR
        if len(detected_text) < 6:
           continue
        if detected_text == "":
            detected_text = "No Text"

        # =========================
        # DRAW BOX
        # =========================
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Text background
        cv2.rectangle(
            frame,
            (x1, y1 - 35),
            (x2, y1),
            (0, 255, 0),
            -1
        )

        # Put text
        cv2.putText(
            frame,
            detected_text,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    # =========================
    # SHOW IMAGE
    # =========================
    cv2.imshow("Image Test", frame)

    print("Press any key for next image")

    cv2.waitKey(0)

# =========================
# CLOSE WINDOWS
# =========================
cv2.destroyAllWindows()