import cv2
import numpy as np
from shape_detection import process_video_frame
import time

def record_demo_video():
    """
    Record a demo video for README file
    """
    # Initialize video capture
    cap = cv2.VideoCapture('http://192.168.137.215:4747/video')
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set video properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('shape_detection_demo.mp4', fourcc, 20.0, (500, 280))
    
    print("Recording demo video...")
    print("Show different shapes to the camera")
    print("Press 'q' to stop recording")
    
    start_time = time.time()
    recording_duration = 15  # Record for 15 seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with shape detection
        processed_frame = process_video_frame(frame)
        
        # Write frame to video (no recording text)
        out.write(processed_frame)
        
        # Display frame
        cv2.imshow('Recording Demo', processed_frame)
        
        # Stop recording after duration or if 'q' is pressed
        elapsed = time.time() - start_time
        if elapsed >= recording_duration or cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Demo video saved as 'shape_detection_demo.mp4'")

if __name__ == "__main__":
    record_demo_video() 