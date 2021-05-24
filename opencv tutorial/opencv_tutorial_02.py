# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
# https://www.pyimagesearch.com/2018/07/19/opencv-tutorial-a-guide-to-learn-opencv/


import imutils
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help='path to input image')
args = vars(ap.parse_args())

img = args['image']

image = cv2.imread(img)

cv2.imshow("Image", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)
cv2.waitKey(0)


edged = cv2.Canny(image, 30, 300) # min_max_threshold
cv2.imshow('Edged', edged)
cv2.waitKey(0)

# Thresholding removes lighter or darker regions and contours of images
# convert pixels greater than 225 to black (0) pixels
# convert pixels less than 225 to 255 (white)
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow('Thresh', thresh)
cv2.waitKey(0)

# Draw contours
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

for c in cnts:
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
    cv2.imshow('Contours', output)
    cv2.waitKey(0)

text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), 2)
cv2.imshow('Text', output)
cv2.waitKey(0)

# EROSION and DILATION
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow('Erosion', mask)
cv2.waitKey(0)

mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow('Dilation', mask)
cv2.waitKey(0)

mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow('Output', output)
cv2.waitKey(0)
