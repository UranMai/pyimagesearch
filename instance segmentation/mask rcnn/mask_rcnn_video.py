import argparse
import time
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", required=True, help="path to input image")  # *mp4
ap.add_argument("-o", "--output", required=True, help="path to output video file")  # *avi
args = vars(ap.parse_args())

labelsPath = '../configs/object_detection_classes_coco.txt'
LABELS = open(labelsPath).read().strip().split('\n')  # 90 classes

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# weights and model configuration definition
weightsPath = '../configs/frozen_inference_graph.pb'
configPath = '../configs/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model=weightsPath, config=configPath)

# ------------------------------------------------------------------
vs = cv2.VideoCapture(args['video'])
writer = None

try:
    prop = cv2.CAP_PROP_FRAME_COUNT  # define property and get it
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
except:
    print("[INFO] could not determine frames in video")
    total = -1

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(['detection_out_final', 'detection_masks'])
    end = time.time()

    for i in range(0, boxes.shape[2]):  # travers top 100 boxes
        # boxes [0, classID, conf, box_four_coords (3:7)]
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > 0.5:
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype('int')
            boxW = endX - startX
            boxH = endY - startY

            mask = masks[i, classID]  # return np.array 15x15
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.3)  # convert to binary array

            roi = frame[startY:endY, startX:endX]
            roi = roi[mask]

            color = COLORS[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype('uint8')
            frame[startY:endY, startX:endX][mask] = blended

            color = [int(c) for c in color]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # four character code, video codecs; MJPG -> avi
            # params: filename, fourcc, fps, size, isColor
            writer = cv2.VideoWriter(args['output'], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
        writer.write(frame)

writer.release()
vs.release()
