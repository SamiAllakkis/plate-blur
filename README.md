# Automatic License Plate Blurring (YOLOv8)

This project detects **license plates** in images and applies a strong blur to them.
It is designed for privacy-preserving image processing and works with JPG, PNG, and WebP images.

The model is trained using **YOLOv8** and applied with **OpenCV**.

---

## 🚀 Features
- Detects full license plates (not just numbers)
- Blurs only the detected plate region
- Supports batch processing of images
- Works on macOS (Apple Silicon), Linux, and Windows

---

## 📦 Requirements

- Python 3.9+
- Ultralytics YOLO
- OpenCV

Install dependencies:
```bash
pip3 install ultralytics opencv-python
