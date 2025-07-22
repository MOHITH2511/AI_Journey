# 🚗 Vehicle Parking & Counting System
A real-time vehicle detection, tracking, and counting system built with YOLOv8, OpenCV, and ByteTrack. This system identifies two-wheelers and four-wheelers, tracks them with unique IDs, and counts them as they cross a customizable line in a video feed.

## Demo
https://github.com/user-attachments/assets/fabb4993-a1a0-41a6-8f61-99826cd856a3

## ✨ Key Features
- **Real-time Vehicle Detection** – Powered by YOLOv8 for fast and accurate results
- **Vehicle Tracking** – Unique ID assignment with ByteTrack
- **Two/Four-Wheeler Counting** – Count vehicles as they cross a defined line
- **Output Video Saving** – Save processed videos with visual overlays
- **Highly Customizable** – Easily configure detection classes, video input/output, and the counting line

## Use Case Scenarios
This model is adaptable for a wide range of smart surveillance and traffic management applications, including:
- **Smart Malls** – Track vehicle inflow/outflow in dynamic parking slots
- **Large Events & Festivals** – Manage crowd vehicle entry in places like temples or stadiums
- **Urban Traffic Monitoring** – Count and analyze vehicle movement in city roads
- **Automated Parking Systems** – Monitor real-time occupancy

## How It Works
1. **Load YOLOv8 Model** – Detects vehicles in each video frame
2. **Frame-by-Frame Processing** – Video is read in real time
3. **Track Using ByteTrack** – Assigns persistent unique IDs across frames
4. **Y-Axis Line Crossing Detection** – Vehicles are counted when they cross a virtual horizontal line
5. **Overlay Results** – Draw bounding boxes, labels, and vehicle counters
6. **Display or Save Output** – Choose between real-time display or saved output video

### Y-Axis Line Adjustment
The system uses a horizontal (Y-axis) counting line to determine when a vehicle has entered or exited a region of interest. You can adjust this line's position in the code by modifying:

```python
COUNT_LINE_Y = 250  # Default line height in pixels
```
📌 You can find and edit this value inside `main.py` or `save_output_video.py`, depending on your use.

Move the line up or down (lower or higher Y-value) depending on your camera angle or video source.

## 🔧 Installation
### Prerequisites
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics (YOLOv8)
- `lap` (for ByteTrack)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/VechileParkingSystem
cd VechileParkingSystem

# Install dependencies
py -m pip install -r requirements.txt
py -m pip install lap
```
- Download YOLOv8 weights (`yolov8n.pt`) from [Ultralytics releases](https://github.com/ultralytics/ultralytics/releases).
- Place your input video in the `inputs/` folder.

## 🚀 Usage
### Real-Time Detection & Counting
```bash
py main.py
```
- A live video window will appear with real-time detections and counters.
- Press 'q' to exit.

## 📁 Project Structure
```
VechileParkingSystem/
├── main.py                # Real-time detection and counting
├── save_output_video.py   # Save processed video
├── requirements.txt       # Dependency list
├── yolov8n.pt             # YOLOv8 weights
├── inputs/                # Input video directory
│   └── entrance_video2.mp4
├── outputs/               # Output video directory
│   └── entrance_video_output2.mp4
└── README.md
```

## ⚙️ Configuration Options
- **Input/Output Paths** – Modify in `main.py` or `save_output_video.py`.
- **Counting Line** – Change `COUNT_LINE_Y` to move the horizontal line.
- **Vehicle Classes** – Customize these lists:
```python
FOUR_WHEELERS = ["car", "bus", "truck"]
TWO_WHEELERS = ["motorcycle", "bicycle"]
```
Add or remove classes as per your scenario.

## 🛠 Troubleshooting
- **Weights Not Found**: Download `yolov8n.pt` from the official Ultralytics page.
- **GUI/Display Issues**: Make sure OpenCV is properly installed with GUI support.
- **Tracking Errors**: Ensure `lap` is installed for ByteTrack to function properly.

---
❤️ **Credits**
Built with passion for smart cities, intelligent transportation, and computer vision solutions.
Contributions are welcome! 
