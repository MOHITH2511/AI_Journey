
# 🧠 Shape Detection System

A real-time computer vision project using **OpenCV** and **Python** that detects basic geometric shapes from both webcam feed and static images. The system uses edge detection and contour analysis to classify shapes like **Circle**, **Rectangle**, **Square**, **Triangle**, and **Hexagon**.

---

## 🎥 Demo

https://github.com/MOHITH2511/AI_Journey/assets/your-video-demo.mp4

*Watch the system detect and label shapes in real-time using color-coded contours and bounding boxes!*

---

## 🧪 Image Processing Pipeline

The shape detection logic involves a sequence of image preprocessing steps before classification:

| Original Image | Grayscale | Blurred | Canny Edges | Final Output |
|----------------|-----------|---------|-------------|--------------|
| ![](output_steps/01_original.jpg) | ![](output_steps/02_gray.jpg) | ![](output_steps/03_blur.jpg) | ![](output_steps/04_canny.jpg) | ![](output_steps/05_final.jpg) |

---

## ✨ Features

- 🟢 Real-time shape detection via webcam
- 🔺 Supports Circles, Triangles, Squares, Rectangles, Hexagons
- 🎨 Color-coded contours per shape
- 📸 Process static images or capture snapshots from video
- 🧠 Modular and readable code structure
- 💾 Auto-save frames and demo video

---

## 📦 Tech Stack

- Python 3.7+
- OpenCV
- NumPy

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/MOHITH2511/AI_Journey.git
cd AI_Journey/shape_detector

# Install dependencies
pip install opencv-python numpy
```

---

## 🚀 Usage

### ▶️ Run Real-time Detection
```bash
python video_capture.py
```
- Press `q` to quit
- Press `s` to save current frame

### 🖼️ Run on Static Images
```bash
python shape_detection.py
```

### 🧪 Test Sample Images
```bash
python sample.py
```

### 🎬 Record Demo Output
```bash
python record_demo.py
```

---

## 📁 Project Structure

```
shape_detector/
├── shape_detection.py         # Shape detection logic
├── video_capture.py           # Live webcam detection
├── sample.py                  # Static image detection
├── record_demo.py             # Record demo output
├── test_images/               # Test input images
│   ├── triangle.png
│   ├── square.png
│   └── ...
├── output_steps/              # Processing step images
│   ├── 01_original.jpg
│   ├── 02_gray.jpg
│   ├── 03_blur.jpg
│   ├── 04_canny.jpg
│   └── 05_final.jpg
├── shape_detection_demo.mp4   # Final demo output (optional)
└── README.md
```

---

## 🔬 How It Works

1. **Resize** and convert input to grayscale  
2. **Apply Gaussian Blur** to reduce noise  
3. **Canny Edge Detection** for identifying strong edges  
4. **Contour Extraction** to find outlines of shapes  
5. **Shape Classification**:
   - Based on number of corners (vertices)
   - Aspect ratio for distinguishing square vs rectangle
6. **Visualization**:
   - Draw colored contours
   - Add center-aligned labels

---

## 🧠 Customization Tips

Inside `shape_detection.py`, you can tweak:

- `PIXELS_TO_MM`: For real-world scaling
- Canny thresholds: `(50, 50)` → try `(100, 150)`
- Minimum contour area: `area_og > 50`
- Add TTS alerts using `pyttsx3` (optional)

---

## 📸 Sample Output

![](output_steps/05_final.jpg)

Shapes are highlighted with custom colors and labeled in real-time.

---

## 🙌 Contributing

Feel free to fork and improve this project!

```bash
1. Fork the repo
2. Create your feature branch
3. Commit your changes
4. Push and open a Pull Request
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for full text.

---

## 👤 Author

**Mohith S**  
[GitHub](https://github.com/MOHITH2511) • [LinkedIn](https://www.linkedin.com/in/mohith-s-954aa52a0/)

> This project is part of my [AI_Journey](https://github.com/MOHITH2511/AI_Journey) — learning AI & Computer Vision through building real projects.

---

**Made with ❤️ and OpenCV by an aspiring CV engineer.**
