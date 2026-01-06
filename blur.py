import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count

# -------------------- SETTINGS --------------------
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
MODEL_PATH = "best.pt"
PLATE_BLUR = (71, 71)
MASK_BLUR = (41, 41)
CONF = 0.3
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp")
MASK_SCALE = 2
# -------------------------------------------------

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

    h, w = img.shape[:2]

    model = YOLO(MODEL_PATH)
    results = model(img, conf=CONF, verbose=False)

    for r in results:
        if r.masks is None:
            continue

        for poly in r.masks.xy:
            pts = np.array(poly, dtype=np.float32)

            # HIGH-RES MASK (ANTI-ALIASED)
            big_mask = np.zeros((h * MASK_SCALE, w * MASK_SCALE), dtype=np.uint8)
            big_pts = (pts * MASK_SCALE).astype(np.int32)

            cv2.fillPoly(
                big_mask,
                [big_pts],
                255,
                lineType=cv2.LINE_AA
            )

            # Smooth edges
            big_mask = cv2.GaussianBlur(big_mask, MASK_BLUR, 0)

            # Downscale smoothly
            mask = cv2.resize(
                big_mask,
                (w, h),
                interpolation=cv2.INTER_AREA
            )

            # Normalize mask
            alpha = mask.astype(np.float32) / 255.0
            alpha = cv2.merge([alpha, alpha, alpha])

            # Blur entire image once per mask
            blurred = cv2.GaussianBlur(img, PLATE_BLUR, 0)

            # Soft blend
            img = (blurred * alpha + img * (1 - alpha)).astype(np.uint8)

    cv2.imwrite(output_path, img)
    print(f"✅ Processed: {filename}")

if __name__ == "__main__":
    start = time.time()

    files = os.listdir(INPUT_DIR)
    workers = max(1, cpu_count() - 1)

    print(f"🚀 Using {workers} processes")

    with Pool(workers) as p:
        p.map(process_image, files)

    elapsed = time.time() - start
    print(f"🎉 Done in {elapsed:.2f} seconds")
