from ultralytics import YOLO
import easyocr
import cv2
import sqlite3
import datetime
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
# DATABASE CONNECTION
# =========================
conn = sqlite3.connect("database/traffic.db")
cursor = conn.cursor()

# =========================
# CREATE EVIDENCE FOLDER
# =========================
os.makedirs("evidence", exist_ok=True)

# =========================
# PROCESS IMAGES
# =========================
for image_name in os.listdir(image_folder):

    image_path = os.path.join(image_folder, image_name)

    print(f"\nProcessing: {image_name}")

    # Read image
    frame = cv2.imread(image_path)

    if frame is None:
        continue

    # Resize
    frame = cv2.resize(frame, (1280, 720))

    # =========================
    # DETECTION
    # =========================
    results = model(frame)[0]

    # =========================
    # LOOP DETECTIONS
    # =========================
    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])

        # Ignore weak detections
        if confidence < 0.4:
            continue

        # Crop vehicle
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

        for result in ocr_results:

            text = result[1]

            # Clean text
            text = re.sub(r'[^A-Z0-9]', '', text.upper())

            detected_text += text + " "

        detected_text = detected_text.strip()

        # Skip empty text
        if detected_text == "":
            continue

        print("Detected:", detected_text)

        # =========================
        # GENERATE DATA
        # =========================
        timestamp = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        violation_type = "Traffic Violation"

        fine_amount = 500

        # =========================
        # SAVE EVIDENCE IMAGE
        # =========================
        evidence_filename = (
            f"{detected_text}_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
        )

        evidence_path = os.path.join(
            "evidence",
            evidence_filename
        )

        cv2.imwrite(evidence_path, vehicle_crop)

        # =========================
        # INSERT INTO DATABASE
        # =========================
        cursor.execute("""
        INSERT INTO violations (
            plate_number,
            violation_type,
            fine_amount,
            timestamp,
            image_path
        )
        VALUES (?, ?, ?, ?, ?)
        """, (
            detected_text,
            violation_type,
            fine_amount,
            timestamp,
            evidence_path
        ))

        conn.commit()

        print("Violation saved to database")

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

        # Display text
        cv2.putText(
            frame,
            detected_text,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    # =========================
    # SHOW IMAGE
    # =========================
    cv2.imshow("Violation Detection", frame)

    print("Press any key for next image")

    cv2.waitKey(0)

# =========================
# CLOSE DATABASE
# =========================
conn.close()

# =========================
# CLOSE WINDOWS
# =========================
cv2.destroyAllWindows()