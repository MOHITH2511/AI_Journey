import cv2
import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(r"D:\AI_Journey\VechileParkingSystem\inputs\entrance_video2.mp4")

COUNT_LINE_Y = 100
BUFFER = 10

two_wheeler_count = 0
four_wheeler_count = 0

# Dictionary to hold tracking history
vehicle_history = {}

# Classes to detect - simplified to just car and motorcycle
FOUR_WHEELERS = ['car']
TWO_WHEELERS = ['motorcycle']

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("✅ Video complete.")
        break

    results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else None
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            label = model.names[cls_id]

            if label not in FOUR_WHEELERS + TWO_WHEELERS or track_id is None:
                continue

            # Initialize if new track_id
            if track_id not in vehicle_history:
                vehicle_history[track_id] = {
                    "label": label,
                    "counted": False,
                    "last_cy": cy
                }

            # Check if it crossed the line (top to bottom)
            last_cy = vehicle_history[track_id]["last_cy"]
            counted = vehicle_history[track_id]["counted"]

            if not counted and last_cy < COUNT_LINE_Y and cy >= COUNT_LINE_Y:
                vehicle_history[track_id]["counted"] = True
                if label in FOUR_WHEELERS:
                    four_wheeler_count += 1
                elif label in TWO_WHEELERS:
                    two_wheeler_count += 1

            # Update last y position
            vehicle_history[track_id]["last_cy"] = cy

            # Draw box and label
            color = (0, 255, 0) if label in FOUR_WHEELERS else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show count
    cv2.putText(frame, f"Two Wheelers: {two_wheeler_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Four Wheelers: {four_wheeler_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Vehicle Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
