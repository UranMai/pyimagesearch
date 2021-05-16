import cv2
import argparse
import imutils
import numpy as np
# https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html

class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		'''
		c - contour argument

		'''
		shape = 'unidetified'
		peri = cv2.arcLength(c, True) # True - whether the curve (c) is closed or not (False)
		approx = cv2.approxPolyDP(c, 0.04*peri, True)
		# return list of vertices

		if len(approx) == 3:
			shape = 'triangle'
		elif len(approx) == 4:
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)

			shape = 'square' if ar >= 0.95 and ar <= 1.05 else 'rectangle'
		elif len(approx) == 5:
			shape = 'pentagon'
		elif len(approx) == 6:
			shape = 'hexagon'
		else:
			shape = 'circle'

		return shape

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the input image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

for c in cnts:
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)

    c = c.astype('float')
    c *= ratio
    c = c.astype('int')
    cv2.drawContours(image, [c], -1, (0, 255,  0), 2)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Image', image)
    cv2.waitKey(0)
	
