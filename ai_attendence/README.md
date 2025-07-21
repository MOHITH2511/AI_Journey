# AI Face Recognition Attendance System

A real-time face recognition system that uses OpenCV for face detection, extracts 512-dimensional embeddings, and compares them with stored embeddings using FAISS for efficient similarity search.

## Features

- **Live Video Feed**: Real-time camera capture and display
- **Face Detection**: Using OpenCV Haar cascades for accurate face detection
- **Face Recognition**: 512-dimensional face embeddings extraction
- **Similarity Search**: FAISS-based efficient similarity matching
- **Real-time Recognition**: Live bounding boxes with matched roll numbers
- **Dataset Management**: Process student images organized by roll numbers

## Project Structure

```
ai_attendence/
├── live_feed.py                    # Basic live video feed
├── face_detector.py               # Face detection using OpenCV
├── face_encoder.py                # Face embedding extraction
├── dataset_processor.py           # Process dataset and build FAISS index
├── face_recognition_system.py     # Complete recognition system
├── live_feed_with_faces.py        # Live feed with face detection
├── requirements.txt               # Python dependencies
└── README.md                     # This file

# Dataset Structure (create this folder)
dataset/
├── 2021001/                      # Student roll number folder
│   ├── student1.jpg
│   ├── student2.jpg
│   └── ...
├── 2021002/                      # Another student
│   ├── student1.jpg
│   └── ...
└── ...
```

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your dataset**:
   - Create a `dataset` folder in the project directory
   - Inside `dataset`, create folders named with student roll numbers
   - Place student images in their respective roll number folders
   - Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

## Usage

### Step 1: Prepare Dataset
Organize your student images as follows:
```
dataset/
├── 2021001/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── 2021002/
│   ├── photo1.jpg
│   └── photo2.jpg
└── 2021003/
    ├── photo1.jpg
    └── photo2.jpg
```

### Step 2: Process Dataset and Build Index
```bash
python dataset_processor.py
```
This will:
- Extract faces from all student images
- Generate 512-dimensional embeddings
- Build a FAISS index for fast similarity search
- Save the index as `face_index.faiss` and metadata as `face_index_metadata.pkl`

### Step 3: Run Face Recognition System
```bash
python face_recognition_system.py
```

## System Controls

### Live Feed Controls
- **Press `q`** - Quit the application
- **Press `s`** - Save current frame
- **Press `f`** - Toggle face detection (in face detection mode)

### Recognition System Controls
- **Press `q`** - Quit the application
- **Press `s`** - Save current frame with recognition results
- **Press `t`** - Adjust similarity threshold (0.0-1.0)

## Testing Individual Components

### Test Basic Live Feed
```bash
python live_feed.py
```

### Test Face Detection
```bash
python face_detector.py
```

### Test Face Encoding
```bash
python face_encoder.py
```

### Test Live Feed with Face Detection
```bash
python live_feed_with_faces.py
```

## Recognition Results

The system will display:
- **Green bounding boxes** with roll numbers for recognized students
- **Red bounding boxes** with "Unknown" for unrecognized faces
- **Confidence scores** for each recognition
- **Real-time FPS** and face count
- **Similarity threshold** setting

## Performance Tips

1. **Image Quality**: Use clear, well-lit images for better recognition
2. **Multiple Images**: Include 3-5 images per student for better accuracy
3. **Threshold Adjustment**: Use the 't' key to adjust similarity threshold:
   - Higher threshold (0.8-0.9): More strict, fewer false positives
   - Lower threshold (0.4-0.6): More lenient, more matches
4. **Lighting**: Ensure good lighting for live recognition

## Troubleshooting

### Common Issues

1. **"No face detected" messages**:
   - Ensure images contain clear, front-facing faces
   - Check image quality and lighting
   - Try different images for the same student

2. **Low recognition accuracy**:
   - Adjust similarity threshold using 't' key
   - Add more diverse images per student
   - Ensure good lighting during live recognition

3. **Camera not working**:
   - Check if camera is connected and not in use by other applications
   - Try different camera index (0, 1, 2, etc.)

4. **FAISS index not found**:
   - Run `dataset_processor.py` first to create the index
   - Ensure `dataset` folder exists with student images

## Technical Details

- **Face Detection**: OpenCV Haar cascades
- **Face Encoding**: ORB features (512-dimensional)
- **Similarity Search**: FAISS with cosine similarity
- **Real-time Processing**: Optimized for 30 FPS
- **Memory Efficient**: FAISS index for fast similarity search

## Requirements

- Python 3.8+
- Webcam
- Windows/Linux/macOS
- Student images organized by roll numbers

## Dependencies

- OpenCV (cv2) - Video capture and image processing
- NumPy - Numerical computations
- FAISS - Similarity search
- scikit-learn - Machine learning utilities
- Matplotlib - Visualization
- Pillow - Image processing 