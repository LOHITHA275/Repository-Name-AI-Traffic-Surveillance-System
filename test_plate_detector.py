from ultralytics import YOLO
import easyocr
import cv2
import re

# =========================
# LOAD MODEL
# =========================
model = YOLO(
    "runs/detect/license_plate_detector_fast-2/weights/best.pt"
)

# =========================
# OCR
# =========================
reader = easyocr.Reader(
    ['en'],
    gpu=False
)

# =========================
# LOAD IMAGE
# =========================
image = cv2.imread(
    r"C:\Users\poorn\OneDrive\Documents\AI-Traffic-Surveillance-System\datasets\license_plate\test\images\Cars77_png.rf.1f1bd61f39d71084856c45af0a614702.jpg"
)

if image is None:
    print("IMAGE NOT FOUND")
    exit()

# =========================
# DETECTION
# =========================
results = model(image)[0]

print("Detections:", len(results.boxes))

for box in results.boxes:

    confidence = float(box.conf[0])

    if confidence < 0.4:
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    plate_crop = image[y1:y2, x1:x2]

    if plate_crop.size == 0:
        continue

    print("Saving plate image...")

    success = cv2.imwrite(
        "debug_plate.jpg",
        plate_crop
    )

    print("Saved:", success)

    # =========================
    # OCR
    # =========================
    ocr_results = reader.readtext(
        plate_crop,
        detail=0
    )

    print("RAW OCR:", ocr_results)

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

    # Common correction for this dataset
    if detected_text.startswith("IR"):
        detected_text = "HR" + detected_text[2:]

    if detected_text == "":
        detected_text = "NO TEXT"

    print("Detected Plate:", detected_text)

    import sqlite3
    from datetime import datetime

    conn = sqlite3.connect("traffic.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detected_plates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number TEXT,
        detected_time TEXT
    )
    """)

    cursor.execute("""
    INSERT INTO detected_plates
    (
        plate_number,
        detected_time
    )
    VALUES (?, ?)
    """,
    (
        detected_text,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

    print("Saved to database")

    # =========================
    # DRAW RESULT
    # =========================
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.rectangle(
        image,
        (x1, y1 - 35),
        (x2, y1),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        image,
        detected_text,
        (x1 + 5, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

cv2.imshow(
    "Plate Detection + OCR",
    image
)

cv2.waitKey(0)
cv2.destroyAllWindows()