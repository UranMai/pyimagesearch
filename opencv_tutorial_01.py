# GUIDE
# https://www.pyimagesearch.com/2018/07/19/opencv-tutorial-a-guide-to-learn-opencv/
# PLAN
# https://www.pyimagesearch.com/start-here/
# COLAB
# https://colab.research.google.com/drive/13IN2dZENZRB1Ta4MXxHcT1LuZeGJkjon?authuser=1#scrollTo=GNQTey7hCgD2


import imutils
import cv2
import argparse
# from google.colab.patches import cv2_imshow
# In Colab use cv2_imshow instead of cv2.imshow

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help='path to input image')
args = vars(ap.parse_args())

img = args['image']
# img = 'images/jp.jpeg'

# OpenCV reads images in BGR ordering
image = cv2.imread(img)
h, w, d = image.shape
print(f'Image {img} has width={w}, height={h}, depth={d}')

cv2.imshow('Image', image)
cv2.waitKey(0) 
#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)


### Array slicing and Cropping ###
# Apply method to auto define ROIs from
# https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
roi = image[40:80, 70:120]
cv2.imshow('ROI', roi)
cv2.waitKey(0)



ratio = 300.0 / w
dim = (300, int(ratio * h))
resized = cv2.resize(image, dim)

resized = imutils.resize(image, width=100)
cv2.imshow('Resized', resized)
cv2.waitKey(0)

center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))

rotated = imutils.rotate(image, -45)

# How it works
# https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
rotated = imutils.rotate_bound(image, 45)

cv2.imshow('Rotated', rotated)
cv2.waitKey(0)

# Smooth image
# blur image to reduce high-frequency noise
# https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
blurred = cv2.GaussianBlur(image, (7, 7), 0)
cv2.imshow('blurred', blurred)
cv2.waitKey(0)

# DRAWING
output = image.copy()
# top-left, bottom-right, color, thickness
cv2.rectangle(output, (70, 40), (120, 90), (0, 0, 255), 2)
cv2.imshow('Rectangle', output)
cv2.waitKey(0)

output = image.copy()
cv2.circle(output, (150, 70), 20, (255, 0, 0), -1)
cv2.imshow('Circle', output)
cv2.waitKey(0)

output = image.copy()
cv2.line(output, (70, 40), (120, 90), (0, 0, 255), 2)
cv2.imshow('Line', output)
cv2.waitKey(0)

output = image.copy()
cv2.putText(output, 'Jurassic Park', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imshow('Text', output)
cv2.waitKey(0)

cv2.destroyAllWindows()

