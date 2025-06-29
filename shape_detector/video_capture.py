import cv2
import numpy as np
from shape_detection import process_video_frame

def main():
    """
    Main function to run live camera shape detection using shape_detection.py
    """
    # Initialize video capture (0 for default camera, you can change to 1, 2, etc. for other cameras)
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Live Shape Detection Started!")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame from camera")
            break
        
        # Use the improved shape detection function from shape_detection.py
        processed_frame = process_video_frame(frame)
        
        # Add frame counter and instructions
        cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(processed_frame, "Press 'q' to quit, 's' to save", (10, processed_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the resulting frame
        cv2.imshow('Live Shape Detection', processed_frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # If 'q' is pressed, break the loop
        if key == ord('q'):
            print("Quitting...")
            break
        # If 's' is pressed, save the current frame
        elif key == ord('s'):
            filename = f"captured_frame_{frame_count}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"Frame saved as {filename}")
        
        frame_count += 1
    
    # Release everything when job is finished
    cap.release()
    cv2.destroyAllWindows()
    print("Live Shape Detection Stopped!")

if __name__ == "__main__":
    main() 