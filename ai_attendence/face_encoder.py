import cv2
import numpy as np
import os
import pickle
from face_detector import FaceDetector

class FaceEncoder:
    def __init__(self, model_path=None):
        """
        Initialize face encoder using OpenCV DNN
        
        Args:
            model_path (str): Path to the face recognition model
        """
        self.face_detector = FaceDetector()
        self.embedding_size = 512
        
        # Try to load a pre-trained face recognition model
        if model_path and os.path.exists(model_path):
            self.model = cv2.dnn.readNetFromCaffe(model_path)
        else:
            # Use a simple approach with OpenCV's face recognition
            print("Using OpenCV-based face encoding (simplified approach)")
            self.model = None
    
    def preprocess_face(self, face_image, target_size=(160, 160)):
        """
        Preprocess face image for encoding
        
        Args:
            face_image: Input face image
            target_size: Target size for the face image
            
        Returns:
            preprocessed_image: Preprocessed face image
        """
        # Resize to target size
        face_resized = cv2.resize(face_image, target_size)

        # Ensure 3 channels
        if len(face_resized.shape) == 2:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2BGR)
        elif len(face_resized.shape) == 3 and face_resized.shape[2] == 1:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2BGR)
        elif len(face_resized.shape) == 3 and face_resized.shape[2] == 4:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGRA2BGR)

        # Convert to float32 and normalize
        face_float = face_resized.astype(np.float32) / 255.0

        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_float, cv2.COLOR_BGR2RGB)

        return face_rgb
    
    def extract_embedding(self, face_image):
        """
        Extract face embedding from face image
        
        Args:
            face_image: Input face image
            
        Returns:
            embedding: 512-dimensional face embedding
        """
        # Ensure face_image is 3-channel BGR
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
        elif len(face_image.shape) == 3 and face_image.shape[2] == 1:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
        elif len(face_image.shape) == 3 and face_image.shape[2] == 4:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2BGR)

        if self.model is None:
            # Simple embedding extraction using image features
            # This is a simplified approach - in production, use a proper face recognition model
            face_processed = self.preprocess_face(face_image)
            
            # Ensure we have a 3-channel image for RGB conversion
            if len(face_processed.shape) == 3 and face_processed.shape[2] == 3:
                # Convert to grayscale for feature extraction
                gray = cv2.cvtColor(face_processed, cv2.COLOR_RGB2GRAY)
            else:
                # If already grayscale or single channel, use as is
                gray = face_processed if len(face_processed.shape) == 2 else cv2.cvtColor(face_processed, cv2.COLOR_BGR2GRAY)
            
            # Extract features using SIFT or ORB
            orb = cv2.ORB_create(nfeatures=512)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            if descriptors is not None and len(descriptors) > 0:
                # Pad or truncate to 512 dimensions
                if descriptors.shape[1] >= 512:
                    embedding = descriptors[0, :512]
                else:
                    embedding = np.zeros(512)
                    embedding[:descriptors.shape[1]] = descriptors[0, :]
            else:
                embedding = np.zeros(512)
            
            # Normalize the embedding
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
        else:
            # Use the loaded model for embedding extraction
            face_processed = self.preprocess_face(face_image)
            blob = cv2.dnn.blobFromImage(face_processed, 1.0, (160, 160), (0, 0, 0), swapRB=True, crop=False)
            self.model.setInput(blob)
            embedding = self.model.forward()
            return embedding.flatten().astype(np.float32)
    
    def extract_face_embedding_from_frame(self, frame):
        """
        Extract face embedding from a frame (detect face first, then encode)
        
        Args:
            frame: Input frame
            
        Returns:
            embeddings: List of face embeddings
            face_boxes: List of face bounding boxes
        """
        # Detect faces in the frame
        face_boxes = self.face_detector.detect_faces(frame)
        embeddings = []
        
        for face_box in face_boxes:
            # Extract face region
            face_region = self.face_detector.extract_face_region(frame, face_box)
            
            # Extract embedding
            embedding = self.extract_embedding(face_region)
            embeddings.append(embedding)
        
        return embeddings, face_boxes

def test_face_encoding():
    """Test function for face encoding"""
    encoder = FaceEncoder()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Face encoding test started. Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Extract face embeddings
        embeddings, face_boxes = encoder.extract_face_embedding_from_frame(frame)
        
        # Draw faces and show embedding info
        for i, (embedding, face_box) in enumerate(zip(embeddings, face_boxes)):
            x, y, w, h = face_box
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Show embedding info
            embedding_norm = np.linalg.norm(embedding)
            cv2.putText(frame, f'Face {i+1}: {embedding_norm:.3f}', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add info text
        cv2.putText(frame, f'Faces: {len(embeddings)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Face Encoding Test', frame)
        
        # Handle key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Face encoding test ended")

if __name__ == "__main__":
    test_face_encoding() 