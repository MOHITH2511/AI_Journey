import cv2
import numpy as np

PIXELS_TO_MM = 0.2645833333  # Adjust as needed for your camera

# Define colors for each shape
SHAPE_COLORS = {
	"Circle": (0, 140, 255),      # Orange
	"Hexagon": (255, 0, 255),    # Magenta
	"Square": (255, 0, 0),       # Blue
	"Rectangle": (0, 255, 0),    # Green
	"Triangle": (0, 255, 255),   # Yellow
	"TBD": (128, 128, 128)       # Gray
}

def detect_shapes_in_image(img):
	"""
	Detect shapes in a given image using improved logic with bounding boxes and labels
	Returns the processed image with shape labels
	"""
	imgResize = cv2.resize(img, (500, 280))
	imgContour = imgResize.copy()

	imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
	imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
	imgCanny = cv2.Canny(imgBlur, 50, 50)

	contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		area_og = int(cv2.contourArea(cnt))
		if area_og > 50:
			approx_points = cv2.approxPolyDP(cnt, 0.02 * (cv2.arcLength(cnt, True)), True)
			object_corners = len(approx_points)
			x, y, w, h = cv2.boundingRect(approx_points)

			# Shape classification
			if object_corners > 7:
				objectType = "Circle"
			elif object_corners == 6:
				objectType = "Hexagon"
			elif object_corners == 4:
				aspRatio = w / float(h)
				if 0.9 < aspRatio < 1.2:
					objectType = "Square"
				else:
					objectType = "Rectangle"
			elif object_corners == 3:
				objectType = "Triangle"
			else:
				objectType = "Unknown"

			color = SHAPE_COLORS.get(objectType, (128, 128, 128))

			# Draw contour in shape-specific color
			cv2.drawContours(imgContour, [cnt], -1, color, 3)
			# Center the label inside the bounding box, in the same color
			label = objectType
			(text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
			label_x = x + (w - text_width) // 2
			label_y = y + (h + text_height) // 2
			cv2.putText(imgContour, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
	return imgContour

def process_video_frame(frame):
	"""
	Process a single video frame for shape detection
	This function is specifically designed for video processing
	"""
	return detect_shapes_in_image(frame)

def main():
	"""
	Original main function for processing a single image file
	"""
	img = cv2.imread(r"D:\AI_Journey\shape_detector\test_images\heptagon.png")

	# Process the image using the extracted function
	processed_img = detect_shapes_in_image(img)

	# Finally show the processed image
	cv2.imshow("Shape Detection", processed_img)
		
	# Closing protocol
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()