from ultralytics import YOLO
import easyocr
import cv2
import re
import sqlite3
from datetime import datetime

# =========================
# DATABASE
# =========================
conn = sqlite3.connect("../database/traffic.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS violations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT,
    violation_type TEXT,
    date_time TEXT
)
""")

conn.commit()

# =========================
# LOAD MODEL
# =========================
model = YOLO(
    "runs/detect/license_plate_detector_fast-2/weights/best.pt"
)

# =========================
# OCR
# =========================
reader = easyocr.Reader(['en'], gpu=False)

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture("videos/plate_test.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()

frame_count = 0
saved_plates = set()

# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended")
        break

    frame_count += 1

    # Process every 3rd frame
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1600, 900))

    # =========================
    # DETECTION
    # =========================
    results = model(frame)[0]

    print("Detections:", len(results.boxes))

    for box in results.boxes:

        confidence = float(box.conf[0])

        if confidence < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # =========================
        # CROP PLATE
        # =========================
        plate_crop = frame[y1:y2, x1:x2]

        if plate_crop.size == 0:
            continue

        cv2.imwrite("debug_plate.jpg", plate_crop)

        # =========================
        # PREPROCESSING
        # =========================
        gray = cv2.cvtColor(
            plate_crop,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        cv2.imwrite("debug_plate_large.jpg", gray)

        # =========================
        # OCR
        # =========================
        try:

            ocr_results = reader.readtext(
                plate_crop,
                detail=0
            )

            print("RAW OCR:", ocr_results)

        except Exception as e:

            print("OCR Error:", e)
            continue

        # =========================
        # TEXT CLEANING
        # =========================
        detected_text = ""

        for text in ocr_results:

            text = re.sub(
                r'[^A-Z0-9]',
                '',
                text.upper()
            )

            detected_text += text

        detected_text = detected_text.strip()

        # Common correction for your dataset
        if detected_text.startswith("IR"):
            detected_text = "HR" + detected_text[2:]

        print("Detected Plate:", detected_text)
        print("Attempting database insert...")

        if len(detected_text) < 6:
            continue

        # =========================
        # SAVE DATABASE
        # =========================
        if detected_text not in saved_plates:

            cursor.execute("""
            INSERT INTO violations
            (
                plate_number,
                violation_type,
                date_time
            )
            VALUES (?, ?, ?)
            """,
            (
                detected_text,
                "Plate Detected",
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ))

            conn.commit()

            saved_plates.add(detected_text)

            print("Saved to database")

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

        cv2.rectangle(
            frame,
            (x1, y1 - 40),
            (x2, y1),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            detected_text,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    cv2.imshow(
        "ANPR System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
conn.close()
cv2.destroyAllWindows()