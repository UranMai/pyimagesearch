import numpy as np
import argparse
import random
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

visualize = -1
labelsPath = '../configs/object_detection_classes_coco.txt'
LABELS = open(labelsPath).read().strip().split('\n')  # 90 classes

colorsPath = 'colors.txt'  # to color objects
COLORS = open(colorsPath).read().strip().split('\n')
COLORS = [np.array(c.split(',')).astype('int') for c in COLORS]
COLORS = np.array(COLORS, dtype='uint8')

# weights and model configuration definition
weightsPath = '../configs/frozen_inference_graph.pb'
configPath = '../configs/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'

net = cv2.dnn.readNetFromTensorflow(model=weightsPath, config=configPath)

image = cv2.imread(args['image'])
(H, W) = image.shape[:2]  # for later scaling of detected objects

blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
(boxes, masks) = net.forward(['detection_out_final', 'detection_masks'])

# top 100 detection boxes [100, 90, 15, 15]
print("[INFO] boxes shape: {}".format(boxes.shape))
print("[INFO] masks shape: {}".format(masks.shape))

for i in range(0, boxes.shape[2]):  # travers top 100 boxes
    # boxes [0, classID, conf, box_four_coords (3:7)]
    classID = int(boxes[0, 0, i, 1])
    confidence = boxes[0, 0, i, 2]

    if confidence > 0.5:
        clone = image.copy()
        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
        (startX, startY, endX, endY) = box.astype('int')
        boxW = endX - startX
        boxH = endY - startY

        mask = masks[i, classID]  # return np.array 15x15
        mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.3)  # convert to binary array

        roi = clone[startY:endY, startX:endX]

        if visualize > 0:
            visMask = (mask * 255).astype('uint8')
            instance = cv2.bitwise_and(roi, roi, mask=visMask)

            cv2.imshow('ROI', roi)
            cv2.imshow('Mask', visMask)
            cv2.imshow('Segmented', instance)

        roi = roi[mask]
        color = random.choice(COLORS)
        blended = ((0.4 * color) + (0.6 * roi)).astype('uint8')
        clone[startY:endY, startX:endX][mask] = blended
        # image[startY:endY, startX:endX][mask] = blended

        color = [int(c) for c in color]
        cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)
        # cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        text = "{}: {:.4f}".format(LABELS[classID], confidence)
        cv2.putText(clone, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # cv2.putText(image, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("Output", clone)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

# cv2.imshow("Output", image)
# cv2.imwrite('image1_segm.jpeg', image)
# key = cv2.waitKey(0)