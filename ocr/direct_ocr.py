import easyocr
import cv2
import os
import re

# =========================
# INITIALIZE OCR
# =========================
reader = easyocr.Reader(['en'])

# =========================
# IMAGE FOLDER
# =========================
image_folder = "videos/plate_pic/images"

# =========================
# LOOP IMAGES
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
    # PREPROCESSING
    # =========================
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Noise reduction
    gray = cv2.bilateralFilter(
        gray,
        11,
        17,
        17
    )

    # Threshold
    gray = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY
    )[1]

    # =========================
    # OCR
    # =========================
    results = reader.readtext(gray)

    detected_text = ""

    # =========================
    # EXTRACT TEXT
    # =========================
    for result in results:

        text = result[1]

        # Clean text
        text = re.sub(
            r'[^A-Z0-9]',
            '',
            text.upper()
        )

        # Keep realistic plates
        if 6 <= len(text) <= 12:
            detected_text += text + " "

            # Bounding box
            points = result[0]

            x1 = int(points[0][0])
            y1 = int(points[0][1])

            x2 = int(points[2][0])
            y2 = int(points[2][1])

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    detected_text = detected_text.strip()

    print("Detected Plate:", detected_text)

    # =========================
    # DISPLAY TEXT
    # =========================
    cv2.putText(
        frame,
        detected_text,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    # =========================
    # SHOW IMAGE
    # =========================
    cv2.imshow("Direct OCR", frame)

    print("Press any key for next image")

    cv2.waitKey(0)

cv2.destroyAllWindows()