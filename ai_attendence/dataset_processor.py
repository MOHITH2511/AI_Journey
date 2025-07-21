import cv2
import numpy as np
import os
import pickle
import faiss
from face_encoder import FaceEncoder
from face_detector import FaceDetector

class DatasetProcessor:
    def __init__(self, dataset_path="dataset"):
        """
        Initialize dataset processor
        
        Args:
            dataset_path (str): Path to the dataset folder containing student images
        """
        self.dataset_path = dataset_path
        self.face_encoder = FaceEncoder()
        self.face_detector = FaceDetector()
        
        # FAISS index for similarity search
        self.index = None
        self.roll_numbers = []  # List to store roll numbers in order
        self.embeddings_list = []  # List to store all embeddings
        
    def load_image(self, image_path):
        """
        Load and preprocess image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            image: Loaded image or None if failed
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def extract_face_from_image(self, image):
        """
        Extract face from image
        
        Args:
            image: Input image
            
        Returns:
            face_image: Extracted face image or None if no face detected
        """
        # Detect faces in the image
        faces = self.face_detector.detect_faces(image)
        
        if len(faces) == 0:
            return None
        
        # Use the largest face (assuming it's the main subject)
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        face_region = self.face_detector.extract_face_region(image, largest_face)
        
        return face_region
    
    def process_student_folder(self, roll_number, student_folder_path):
        """
        Process all images in a student's folder
        
        Args:
            roll_number (str): Student's roll number
            student_folder_path (str): Path to student's image folder
            
        Returns:
            embeddings: List of embeddings for this student
        """
        embeddings = []
        image_files = [f for f in os.listdir(student_folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"Processing {len(image_files)} images for roll number: {roll_number}")
        
        for image_file in image_files:
            image_path = os.path.join(student_folder_path, image_file)
            image = self.load_image(image_path)
            
            if image is None:
                continue
            
            # Extract face from image
            face_image = self.extract_face_from_image(image)
            
            if face_image is None:
                print(f"No face detected in {image_file}")
                continue
            
            # Debug: print face region shape and dtype
            print(f"  - Face region from {image_file}: shape={face_image.shape}, dtype={face_image.dtype}")
            
            # Extract embedding
            try:
                embedding = self.face_encoder.extract_embedding(face_image)
                embeddings.append(embedding)
                print(f"  ✓ Extracted embedding from {image_file}")
            except Exception as e:
                print(f"  ✗ Failed to extract embedding from {image_file}: {e}")
        
        return embeddings
    
    def process_dataset(self):
        """
        Process the entire dataset and build FAISS index
        
        Returns:
            success (bool): True if processing was successful
        """
        if not os.path.exists(self.dataset_path):
            print(f"Dataset path does not exist: {self.dataset_path}")
            return False
        
        print(f"Processing dataset from: {self.dataset_path}")
        print("=" * 50)
        
        # Get all student folders (roll numbers)
        student_folders = [f for f in os.listdir(self.dataset_path) 
                          if os.path.isdir(os.path.join(self.dataset_path, f))]
        
        if len(student_folders) == 0:
            print("No student folders found in dataset")
            return False
        
        print(f"Found {len(student_folders)} student folders")
        
        # Process each student folder
        for roll_number in sorted(student_folders):
            student_folder_path = os.path.join(self.dataset_path, roll_number)
            
            # Extract embeddings for this student
            embeddings = self.process_student_folder(roll_number, student_folder_path)
            
            if len(embeddings) > 0:
                # Add embeddings to the list
                for embedding in embeddings:
                    self.embeddings_list.append(embedding)
                    self.roll_numbers.append(roll_number)
                
                print(f"Added {len(embeddings)} embeddings for roll number: {roll_number}")
            else:
                print(f"No valid embeddings found for roll number: {roll_number}")
        
        # Build FAISS index
        if len(self.embeddings_list) > 0:
            self.build_faiss_index()
            print(f"\nSuccessfully processed {len(self.embeddings_list)} embeddings from {len(set(self.roll_numbers))} students")
            return True
        else:
            print("No embeddings were extracted from the dataset")
            return False
    
    def build_faiss_index(self):
        """Build FAISS index from extracted embeddings"""
        if len(self.embeddings_list) == 0:
            print("No embeddings to build index from")
            return
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(self.embeddings_list, dtype=np.float32)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]  # Should be 512
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add embeddings to index
        self.index.add(embeddings_array)
        
        print(f"Built FAISS index with {self.index.ntotal} embeddings of dimension {dimension}")
    
    def save_index(self, output_path="face_index"):
        """
        Save FAISS index and metadata
        
        Args:
            output_path (str): Base path for saving index files
        """
        if self.index is None:
            print("No index to save")
            return
        
        # Save FAISS index
        faiss.write_index(self.index, f"{output_path}.faiss")
        
        # Save metadata
        metadata = {
            'roll_numbers': self.roll_numbers,
            'total_embeddings': len(self.embeddings_list),
            'unique_students': len(set(self.roll_numbers))
        }
        
        with open(f"{output_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved index to {output_path}.faiss")
        print(f"Saved metadata to {output_path}_metadata.pkl")
    
    def load_index(self, index_path="face_index"):
        """
        Load FAISS index and metadata
        
        Args:
            index_path (str): Base path for loading index files
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{index_path}.faiss")
            
            # Load metadata
            with open(f"{index_path}_metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.roll_numbers = metadata['roll_numbers']
            
            print(f"Loaded index with {self.index.ntotal} embeddings")
            print(f"Loaded metadata for {metadata['unique_students']} unique students")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def search_similar_faces(self, query_embedding, k=5, threshold=0.5):
        """
        Search for similar faces in the index
        
        Args:
            query_embedding: Query face embedding
            k (int): Number of top results to return
            threshold (float): Similarity threshold
            
        Returns:
            results: List of (roll_number, similarity_score) tuples
        """
        if self.index is None:
            print("No index loaded")
            return []
        
        # Reshape query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search in FAISS index
        similarities, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.roll_numbers) and similarity >= threshold:
                roll_number = self.roll_numbers[idx]
                results.append((roll_number, float(similarity)))
        
        return results

def main():
    """Main function to process dataset"""
    print("Dataset Processor for Face Recognition System")
    print("=" * 50)
    
    # Initialize processor
    processor = DatasetProcessor()
    
    # Process dataset
    if processor.process_dataset():
        # Save index
        processor.save_index()
        print("\nDataset processing completed successfully!")
    else:
        print("\nDataset processing failed!")

if __name__ == "__main__":
    main() 