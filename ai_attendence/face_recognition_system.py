import cv2
import numpy as np
import time
from face_detector import FaceDetector
from face_encoder import FaceEncoder
from dataset_processor import DatasetProcessor

class FaceRecognitionSystem:
    def __init__(self, index_path="face_index"):
        """
        Initialize the complete face recognition system
        
        Args:
            index_path (str): Path to the FAISS index files
        """
        self.face_detector = FaceDetector()
        self.face_encoder = FaceEncoder()
        self.dataset_processor = DatasetProcessor()
        
        # Load the pre-built index
        self.index_loaded = self.dataset_processor.load_index(index_path)
        
        if not self.index_loaded:
            print("Warning: Could not load face index. Please run dataset_processor.py first.")
        
        # Recognition parameters
        self.similarity_threshold = 0.6
        self.recognition_history = {}  # Store recent recognitions to avoid flickering
        
    def start_camera(self, camera_index=0):
        """Start the camera capture"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera at index {camera_index}")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera started successfully at index {camera_index}")
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera capture"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("Camera stopped")
    
    def get_frame(self):
        """Get a single frame from the camera"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def recognize_faces(self, frame):
        """
        Recognize faces in the frame
        
        Args:
            frame: Input frame
            
        Returns:
            frame: Frame with recognition results
            recognition_results: List of (face_box, roll_number, confidence) tuples
        """
        if not self.index_loaded:
            return frame, []
        
        # Extract face embeddings from frame
        embeddings, face_boxes = self.face_encoder.extract_face_embedding_from_frame(frame)
        
        recognition_results = []
        
        for i, (embedding, face_box) in enumerate(zip(embeddings, face_boxes)):
            # Search for similar faces in the index
            results = self.dataset_processor.search_similar_faces(
                embedding, k=1, threshold=self.similarity_threshold
            )
            
            if results:
                roll_number, confidence = results[0]
                recognition_results.append((face_box, roll_number, confidence))
            else:
                recognition_results.append((face_box, "Unknown", 0.0))
        
        return frame, recognition_results
    
    def draw_recognition_results(self, frame, recognition_results):
        """
        Draw recognition results on the frame
        
        Args:
            frame: Input frame
            recognition_results: List of (face_box, roll_number, confidence) tuples
            
        Returns:
            frame: Frame with recognition results drawn
        """
        for face_box, roll_number, confidence in recognition_results:
            x, y, w, h = face_box
            
            # Choose color based on recognition result
            if roll_number == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                label = "Unknown"
            else:
                color = (0, 255, 0)  # Green for recognized
                label = f"{roll_number} ({confidence:.2f})"
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run_recognition_system(self):
        """Run the complete face recognition system"""
        if not self.start_camera():
            return
        
        print("Face Recognition System Started")
        print("Controls: Press 'q' to quit, 's' to save frame, 't' to adjust threshold")
        print(f"Current similarity threshold: {self.similarity_threshold}")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            frame = self.get_frame()
            if frame is None:
                print("Failed to capture frame")
                break
            
            # Recognize faces in the frame
            frame, recognition_results = self.recognize_faces(frame)
            
            # Draw recognition results
            frame = self.draw_recognition_results(frame, recognition_results)
            
            # Add frame counter and FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Add info overlay
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(recognition_results)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Threshold: {self.similarity_threshold:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show recognized students
            recognized_students = [r[1] for r in recognition_results if r[1] != "Unknown"]
            if recognized_students:
                cv2.putText(frame, f"Recognized: {', '.join(set(recognized_students))}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(frame, "Press 'q' to quit, 's' to save, 't' to adjust threshold", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Face Recognition System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"recognition_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('t'):
                # Adjust threshold
                print(f"Current threshold: {self.similarity_threshold}")
                try:
                    new_threshold = float(input("Enter new threshold (0.0-1.0): "))
                    if 0.0 <= new_threshold <= 1.0:
                        self.similarity_threshold = new_threshold
                        print(f"Threshold updated to: {self.similarity_threshold}")
                    else:
                        print("Invalid threshold value. Must be between 0.0 and 1.0")
                except ValueError:
                    print("Invalid input. Threshold unchanged.")
        
        self.stop_camera()
        cv2.destroyAllWindows()
        print("Face recognition system ended")

def main():
    """Main function to run the face recognition system"""
    print("AI Face Recognition Attendance System")
    print("=" * 50)
    
    # Check if index exists
    import os
    if not os.path.exists("face_index.faiss"):
        print("Face index not found. Please run dataset_processor.py first to create the index.")
        print("Make sure you have a 'dataset' folder with student images organized by roll numbers.")
        return
    
    # Create and run face recognition system
    system = FaceRecognitionSystem()
    system.run_recognition_system()

if __name__ == "__main__":
    main() 