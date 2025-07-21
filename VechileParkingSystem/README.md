# Vehicle Parking System

A real-time vehicle detection, tracking, and counting system using YOLOv8 and OpenCV. Detects and counts two-wheelers and four-wheelers as they cross a defined line in video feeds.

## Demo

*Detect and count vehicles in real-time from video feeds!*

## ✨ Features

- **Real-time Vehicle Detection**: YOLOv8-based detection
- **Vehicle Tracking**: ByteTrack for unique ID assignment
- **Two/Four Wheeler Counting**: Counts as vehicles cross a line
- **Output Video Saving**: Save processed video with overlays
- **Customizable Classes & Line**: Easily adjust vehicle types and counting line

## Installation

### Prerequisites
- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)
- PyTorch
- lap (for tracking)

### Setup
```bash
# Clone the repository
cd VechileParkingSystem

# Install dependencies
py -m pip install -r requirements.txt
py -m pip install lap
```

- Download `yolov8n.pt` weights from [Ultralytics YOLOv8 releases](https://github.com/ultralytics/ultralytics/releases) if not present.
- Place your input video in the `inputs/` directory.

## Usage

### Real-Time Detection & Counting
```bash
py main.py
```
- Shows a window with live detection and counters
- Press `'q'` to quit

### Save Output Video
```bash
py save_output_video.py
```
- Saves processed video with overlays to `outputs/`

## Project Structure
```
VechileParkingSystem/
├── main.py                # Real-time detection and counting
├── save_output_video.py   # Save processed video
├── requirements.txt       # Dependencies
├── yolov8n.pt             # YOLOv8 weights
├── inputs/                # Input videos
│   └── entrance_video2.mp4
├── outputs/               # Output videos
│   └── entrance_video_output2.mp4
└── README.md
```

## 🎯 How It Works

1. **Load YOLOv8 Model**: For vehicle detection
2. **Read Video Frame-by-Frame**
3. **Detect & Track Vehicles**: Assign unique IDs
4. **Count on Line Crossing**: Increment counters when vehicles cross the defined Y-axis line
5. **Draw Results**: Bounding boxes, labels, and counters on each frame
6. **Display or Save**: Show in a window or write to output video

## Configuration
- **Input/Output Paths**: Set in `main.py` or `save_output_video.py`
- **Counting Line**: Adjust `COUNT_LINE_Y` for line position
- **Vehicle Classes**: Edit `FOUR_WHEELERS` and `TWO_WHEELERS` lists

## Troubleshooting
- Ensure all dependencies are installed
- For missing weights, download `yolov8n.pt`
- For GUI issues, check OpenCV installation

---

**Made with ❤️ for smart parking and computer vision applications** 