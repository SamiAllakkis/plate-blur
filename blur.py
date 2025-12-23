import os
import cv2
import time 
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count

start_time = time.time()

# --------------------SETTINGS--------------------
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
MODEL_PATH = "runs/detect/train/weights/best.pt"
BLUR = (99, 99)
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp")
# ------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_image(filename):
    if not filename.lower().endswith(VALID_EXTS):
        return

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    img = cv2.imread(input_path)
    if img is None:
        print(f"⚠️ Skipping unreadable file: {filename}")
        return

    model = YOLO(MODEL_PATH)

    results = model(img)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, BLUR, 0)

    cv2.imwrite(output_path, img)
    print(f"Processed: {filename}")

if __name__ == "__main__":
    start = time.time()

    files = os.listdir(INPUT_DIR)
    workers = max(1, cpu_count() - 1)

    with Pool(workers) as p:
        p.map(process_image, files)

    elapsed = time.time() - start
    print(f"✅ Done in {elapsed:.2f} seconds using {workers} processes")