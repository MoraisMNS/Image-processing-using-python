
# 🧠 EC7212 – Computer Vision & Image Processing
### 🎓 Take-Home Assignment 01

---

## 📦 Project Overview

This repository includes four Python programs that demonstrate key concepts in image processing using OpenCV. Tasks include intensity quantization, spatial filtering, rotation, and resolution reduction.

---

## ⚙️ Setup Instructions

Install the required packages using:

```bash
pip install opencv-python numpy
```

---

## 📂 Assignment Breakdown

### 🔹 Que1.py - Intensity Quantization
- Converts a grayscale image to a reduced number of intensity levels.
- User specifies a power-of-two level (e.g., 2, 4, ..., 256).
- Displays original vs quantized image side-by-side.

### 🔹 Que2.py - Mean Filtering
- Applies mean filters of size 3×3, 10×10, and 20×20.
- Visualizes the effect of different filter sizes in a 2×2 grid.

### 🔹 Que3.py - Image Rotation
- Rotates the image by 45° and 90°.
- Shows original and rotated images in a grid layout.

### 🔹 Que4.py - Block-Based Downsampling
- Reduces spatial resolution using 3×3, 5×5, and 7×7 block averages.
- Useful for simulating low-resolution imaging.

---

## 🖼️ How to Use

1. Run a script:
    ```bash
    python Q01.py
    ```

2. Enter the full path to your image when prompted.

---
