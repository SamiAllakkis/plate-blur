# Automatic License Plate Blurring (YOLOv8)

This project automatically detects vehicle license plates in images and applies a **strong blur** to them for privacy protection.
It is built using **YOLOv8** for detection and **OpenCV** for image processing.

The script supports **JPG, PNG, and WebP** images and can process multiple images in batch.

---

## 🚀 Features
- Detects full license plates (not just characters)
- Applies blur only to the detected plate region
- Supports batch processing
- Preserves original image resolution
- Works on macOS (Apple Silicon & Intel), Linux, and Windows

---

## 🧠 Technologies Used

- YOLOv8 – real-time object detection model (Ultralytics)
- OpenCV – image processing and blurring
- Python – main programming language

---

## 📦 Requirements

- Python 3.9+
- Ultralytics YOLO
- OpenCV
Install dependencies:
```bash
pip3 install ultralytics opencv-python

---
