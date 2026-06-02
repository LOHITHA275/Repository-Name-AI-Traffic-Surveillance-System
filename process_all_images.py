from ultralytics import YOLO
import easyocr
import cv2
import re
import sqlite3
from datetime import datetime
import os

# =========================
# DATABASE
# =========================
conn = sqlite3.connect("traffic.db")
cursor = conn.cursor()

import os

print("Database file:")
print(os.path.abspath("traffic.db"))

cursor.execute("PRAGMA table_info(detected_plates)")
print(cursor.fetchall())

cursor.execute("""
CREATE TABLE IF NOT EXISTS detected_plates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT,
    image_name TEXT,
    detected_time TEXT
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
# TEST IMAGE FOLDER
# =========================
IMAGE_FOLDER = r"../datasets/license_plate/test/images"

saved_plates = set()

total_images = 0
successful_reads = 0

for filename in os.listdir(IMAGE_FOLDER):

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    image_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    total_images += 1

    print(f"\nProcessing: {filename}")

    results = model(image)[0]

    for box in results.boxes:

        confidence = float(box.conf[0])

        if confidence < 0.4:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        plate_crop = image[y1:y2, x1:x2]

        if plate_crop.size == 0:
            continue

        ocr_results = reader.readtext(
            plate_crop,
            detail=0
        )

        detected_text = ""

        for text in ocr_results:

            text = re.sub(
                r'[^A-Z0-9]',
                '',
                text.upper()
            )

            detected_text += text

        detected_text = detected_text.strip()

        if detected_text.startswith("IR"):
            detected_text = "HR" + detected_text[2:]

        if len(detected_text) < 6:
            continue

        print("Plate:", detected_text)

        if detected_text not in saved_plates:

            cursor.execute("""
            INSERT INTO detected_plates
            (
                plate_number,
                image_name,
                detected_time
            )
            VALUES (?, ?, ?)
            """,
            (
                detected_text,
                filename,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ))

            conn.commit()

            saved_plates.add(
                detected_text
            )

            successful_reads += 1

            print("Saved to database")

print("\n======================")
print("Total Images:", total_images)
print("Saved Plates:", successful_reads)
print("======================")

conn.close()