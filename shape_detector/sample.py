import cv2
import numpy as np

PIXELS_TO_MM = 0.2645833333  # Adjust as needed for your camera

def detect_shapes_in_image(img):
	imgResize = cv2.resize(img, (500, 280))
	imgContour = imgResize.copy()

	imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
	imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
	imgCanny = cv2.Canny(imgBlur, 50, 50)

	contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		area_og = int(cv2.contourArea(cnt))
		if area_og > 50:
			cv2.drawContours(imgContour, [cnt], -1, (255, 0, 0), 3)
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
				objectType = "TBD"

			# Draw bounding box
			cv2.rectangle(imgContour, (x-5, y-5), (x+w+2, y+h+2), (0, 255, 0), 2)
			# Center the label inside the bounding box
			label = objectType
			(text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
			label_x = x + (w - text_width) // 2
			label_y = y + (h + text_height) // 2
			cv2.putText(imgContour, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
	return imgContour

if __name__ == "__main__":
	img = cv2.imread(r"D:\AI_Journey\shape_detector\assets\test_images\shapes.png")
	processed_img = detect_shapes_in_image(img)
	cv2.imshow("Shape Detection Test", processed_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()