from __future__ import print_function
# https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np 
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help='path to images dir')
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imagePath in paths.list_images(args['images']):
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()

	# detect people
	(rects, weights) = hog.detectMultiScale(image, winStride=(4,4), padding=(8,8), scale=1.05)

	# draw bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x,y), (x+w, y+h), (0,0,255), 2)

	# apply non-maxima suppression (NMS) to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw final boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA,yA), (xB,yB), (0,255,0),2)

	filename = imagePath[imagePath.rfind('/')+1:]
	print("{} : {} original boxes, {} after suppresion".format(filename, len(rects), len(pick)))

	cv2.imshow('Before NMS', orig)
	cv2.imshow('After NMS', image)
	cv2.waitKey(0)
