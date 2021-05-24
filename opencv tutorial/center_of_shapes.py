# PLAN
# detect outline of each shape
# compute center of the contour (centroid)

# 1. cvt to gray color
# 2. blur to reduce high noise 
# 3. image binarization, edge detection and thresholding

import cv2
import imutils
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help='path to input image')
args = vars(ap.parse_args())

img = args['image']

def center_of_shapes(img):
	image = cv2.imread(img)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

	# How it works
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	for c in cnts:
		# compute the center of the contour
		M = cv2.moments(c)
		cX = int(M["m10"] / (M["m00"]+0.1))
		cY = int(M["m01"] / (M["m00"]+0.1))
		# draw the contour and center of the shape on the image
		cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
		cv2.putText(image, "center", (cX - 20, cY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		# show the image
		cv2.imshow("Image", image)
		cv2.waitKey(0)

# center_of_shapes(img)

def find_shapes(img):
	# https://www.pyimagesearch.com/2014/10/20/finding-shapes-images-using-python-opencv/
	# detect black shapes
	image = cv2.imread(img)
	cv2.imshow('Image', image)
	cv2.waitKey(0)

	lower = np.array([0, 0, 0])
	upper = np.array([15, 15, 15])
	shapeMask = cv2.inRange(image, lower, upper)

	cv2.imshow('Black shapes', shapeMask)
	cv2.waitKey(0)

	# only for binary images 
	cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	print("I found {} black shapes".format(len(cnts)))
	cv2.imshow("Mask", shapeMask)

	for c in cnts:
		# destination image, all imput contours, id of contour to draw, color, thickness, lineType, 
		cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		cv2.imshow('Images', image)
		cv2.waitKey(0)

# find_shapes(img)

def find_extreme_points(img):
	image = cv2.imread(img)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)

	thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# print(c.shape) # 1113, 1, 2, where 2 is x, y coords
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])

	cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
	cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
	cv2.circle(image, extRight, 8, (0, 255, 0), -1)
	cv2.circle(image, extTop, 8, (255, 0, 0), -1)
	cv2.circle(image, extBot, 8, (255, 255, 0), -1)

	cv2.imshow("Image", image)
	cv2.waitKey(0)

# find_extreme_points(img)
# https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
def sort_contours(cnts, method='left-to-right'):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0

	if method=='right-to-left' or method=='bottom-to-top':
		reverse=True

	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1

	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)

	return (cnts, boundingBoxes)

def draw_contour(image, c, i):
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

	cv2.putText(image, '#{}'.format(i+1), (cX-20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
	return image


