from collections import deque
from imutils.video import VideoStream
import numpy as np 
import argparse
import cv2
import imutils
import time
import imageio


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", 
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# If video is supplied, we will track video frames. 
# Otherwise, OpenCV will try to access webcam

# --buffer - max size of deque, the size(amount) of tracked ball locations

# define lower and upper boundaries of "green" ball in HSV color space
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

if not args.get('video', False):
	vs = VideoStream(src=0).start() # access to webcam
else:
	vs = cv2.VideoCapture(args["video"]) # supply video

# allow the camera or video file to warm up
time.sleep(2.0)

frames = []
image_count = 0
while True:
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	if frame is None:
		break

	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and init center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, 
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		if radius > 10:
			cv2.circle(frame, (int(x), int(y)), int(radius), 
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	pts.appendleft(center)

	for i in range(1, len(pts)):
		if pts[i-1] is None or pts[i] is None:
			continue

		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord('a'):
		image_count += 1
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frames.append(rgb_frame)
		print('Adding new image:', image_count)
	elif key == ord('q'):
		break

with imageio.get_writer('ball.gif', mode='I') as writer:
	for idx, frame in enumerate(frames):
		writer.append_data(frame)

if not args.get("video", False):
	vs.stop()
else:
	vs.release()

cv2.destroyAllWindows()
