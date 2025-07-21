import cv2
import numpy as np
import time

class LiveFeed:
    def __init__(self, camera_index=0):
        """
        Initialize the live feed with webcam
        
        Args:
            camera_index (int): Index of the camera to use (default: 0 for primary camera)
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        
    def start_camera(self):
        """Start the camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera at index {self.camera_index}")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera started successfully at index {self.camera_index}")
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera capture"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        print("Camera stopped")
    
    def get_frame(self):
        """Get a single frame from the camera"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def run_live_feed(self):
        """Run the live feed with real-time display"""
        if not self.start_camera():
            return
        
        self.is_running = True
        print("Live feed started. Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            frame = self.get_frame()
            if frame is None:
                print("Failed to capture frame")
                break
            
            # Add frame counter and FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Add text overlay
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save frame", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Live Feed', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"captured_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
        
        self.stop_camera()
        cv2.destroyAllWindows()
        print("Live feed ended")

def main():
    """Main function to run the live feed"""
    print("Starting Face Recognition Live Feed System")
    print("=" * 50)
    
    # Create and run live feed
    live_feed = LiveFeed()
    live_feed.run_live_feed()

if __name__ == "__main__":
    main() 