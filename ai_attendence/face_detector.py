import cv2
import numpy as np
from deepface.detectors import FaceDetector

class FaceDetectorWrapper:
    def __init__(self):
        self.detector_name = "retinaface"
        self.detector, self.detector_input_size = FaceDetector.build_model(self.detector_name)
        print("DeepFace RetinaFace detector initialized successfully")

    def detect_faces(self, frame):
        # DeepFace expects BGR images
        faces = FaceDetector.detect_faces(self.detector, self.detector_name, frame)
        boxes = []
        for face, (x, y, w, h) in faces:
            boxes.append((x, y, w, h))
        return boxes

    def draw_faces(self, frame, faces, color=(0, 255, 0), thickness=2):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(frame, 'Face', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def extract_face_region(self, frame, face_box):
        x, y, w, h = face_box
        face_region = frame[y:y+h, x:x+w]
        return face_region

def test_face_detection():
    import time
    detector = FaceDetectorWrapper()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    print("Face detection test started. Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        faces = detector.detect_faces(frame)
        frame = detector.draw_faces(frame, faces)
        cv2.putText(frame, f'Faces detected: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Face Detection Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Face detection test ended")

if __name__ == "__main__":
    test_face_detection() 