import numpy as np
import cv2

img = cv2.imread(r"D:\AI_Journey\shape_detector\test_images\heptagon.png")


#Converting the image to Gray Scale
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

#Removing Gaussian Noise
blur = cv2.GaussianBlur(gray, (3,3),0)

#Applying inverse binary due to white background and adapting thresholding for better results
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 205, 1)

#Finding contours with simple retrieval (no hierarchy) and simple/compressed end points
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#Checking to see how many contours were found
print(len(contours))

#An empty list to store filtered contours
filtered = []

#Looping over all found contours
for c in contours:
	#If it has significant area, add to list
	if cv2.contourArea(c) < 1000:continue
	filtered.append(c)

#Checking the number of filtered contours
print(len(filtered))

#Initialize an equally shaped image
objects = np.zeros([img.shape[0],img.shape[1],3], 'uint8')

#Looping over filtered contours
for c in filtered:
	# #Select a random color to draw the contour
	# col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
	#Draw the contour on the image with above color
	cv2.drawContours(objects,[c], -1, (255,0,0), -1)
	#Fetch contour area
	area = cv2.contourArea(c)
	#Fetch the perimeter
	p = cv2.arcLength(c,True)
	print(area,p)

	# Approximate the contour to a polygon
	approx = cv2.approxPolyDP(c, 0.02 * p, True)
	sides = len(approx)
	if sides == 3:
		shape = "Triangle"
	elif sides == 4:
		x, y, w, h = cv2.boundingRect(approx)
		aspect_ratio = float(w) / h
		if 0.95 <= aspect_ratio <= 1.05:
			shape = "Square"
		else:
			shape = "Rectangle"
	elif sides == 5:
		shape = "Pentagon"
	elif sides == 6:
		shape = "Hexagon"
	elif sides == 7:
		shape = "Heptagon"
	elif sides == 8:
		shape = "Octagon"
	elif sides > 8:
		shape = "Circle"
	else:
		shape = "Unknown"

	# Put the shape name near the first vertex of the contour
	x, y = approx[0][0]
	cv2.putText(objects, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
	print(f"Detected shape: {shape}")

#Finally show the processed image
cv2.imshow("Contours",objects)
	
#Closing protocol
cv2.waitKey(0)
cv2.destroyAllWindows()