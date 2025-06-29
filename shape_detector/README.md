# Shape Detection System

A real-time shape detection system using OpenCV and Python that can identify various geometric shapes from camera feed or images.

## 🎥 Demo

![Shape Detection Demo](shape_detection_demo.gif)

*Watch the system detect circles, triangles, squares, rectangles, and hexagons in real-time!*

## ✨ Features

- **Real-time Shape Detection**: Detect shapes from live camera feed
- **Multiple Shape Recognition**: 
  - Circles
  - Triangles
  - Squares
  - Rectangles
  - Hexagons
- **Color-coded Output**: Each shape type has its own distinct color
- **Image Processing**: Process static images for shape detection
- **Modular Design**: Separate modules for video capture and shape detection

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- OpenCV
- NumPy

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/shape_detector.git
cd shape_detector

# Install dependencies
pip install opencv-python numpy
```

## 🚀 Usage

### Live Camera Detection
```bash
python video_capture.py
```
- Press `'q'` to quit
- Press `'s'` to save current frame

### Image Processing
```bash
python shape_detection.py
```

### Testing with Sample Images
```bash
python sample.py
```

### Record Demo Video
```bash
python record_demo.py
```

## 📁 Project Structure

```
shape_detector/
├── shape_detection.py      # Main shape detection logic
├── video_capture.py        # Live camera processing
├── sample.py              # Testing with sample images
├── record_demo.py         # Demo video recording
├── test_images/           # Sample images for testing
│   ├── circle.jpg
│   ├── triangle.png
│   ├── square.png
│   └── ...
└── README.md
```

## 🎯 How It Works

1. **Image Preprocessing**: Convert to grayscale, apply Gaussian blur, and edge detection
2. **Contour Detection**: Find contours using Canny edge detection
3. **Shape Classification**: Classify shapes based on vertex count and aspect ratio
4. **Visual Output**: Display results with color-coded contours and labels

## 🔧 Configuration

You can adjust various parameters in `shape_detection.py`:
- `PIXELS_TO_MM`: Scale factor for measurements
- Contour area threshold (currently 50)
- Canny edge detection parameters
- Shape classification thresholds

## 📸 Sample Output

The system provides:
- Color-coded shape contours
- Centered shape labels
- Real-time processing
- Clean, professional visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Python community for excellent libraries
- Contributors and testers

---

**Made with ❤️ for computer vision enthusiasts** 