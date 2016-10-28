import numpy as np
import matplotlib.pyplot as plt
# import Image
import sys
import os
import PIL
import operator
from  math import pow
from PIL import Image, ImageDraw, ImageFont
import cv2
caffe_root = '/home/icog-labs/caffe/'

sys.path.insert(0, caffe_root + 'python')
import caffe

# caffe.set_device(0)
# caffe.set_mode_gpu()



# helper show filter outputs
def show_filters(net):
    net.forward()
    plt.figure()
    filt_min, filt_max = net.blobs['conv'].data.min(), net.blobs['conv'].data.max()
    for i in range(3):  # three feature map.
        plt.subplot(1, 4, i + 2)
        plt.title("filter #{} output".format(i))
        plt.imshow(net.blobs['conv'].data[0, i], vmin=filt_min, vmax=filt_max)
        plt.tight_layout()
        plt.axis('off')
        plt.show()


def generateBoundingBox(featureMap, scale):
    boundingBox = []
    stride = 32
    cellSize = 227
    # 227 x 227 cell, stride=32
    for (x, y), prob in np.ndenumerate(featureMap):
        if (prob >= 0.85):
            boundingBox.append(
                [float(stride * y) / scale, float(x * stride) / scale, float(stride * y + cellSize - 1) / scale,
                 float(stride * x + cellSize - 1) / scale, prob])
    # sort by prob, from max to min.
    # boxes = np.array(boundingBox)
    return boundingBox


def nms_average(boxes, overlapThresh=0.2):
    result_boxes = []
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:, 4])

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # area of i.
        area_i = np.maximum(0, x2[i] - x1[i] + 1) * np.maximum(0, y2[i] - y1[i] + 1)
        area_array = np.zeros(len(idxs) - 1)
        area_array.fill(area_i)
        # compute the ratio of overlap
        # overlap = (w * h) / (area[idxs[:last]]  - w * h + area_array)

        overlap = (w * h) / (area[idxs[:last]])
        delete_idxs = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        xmin = 10000
        ymin = 10000
        xmax = 0
        ymax = 0
        ave_prob = 0
        width = x2[i] - x1[i] + 1
        height = y2[i] - y1[i] + 1
        for idx in delete_idxs:
            ave_prob += boxes[idxs[idx]][4]
            if (boxes[idxs[idx]][0] < xmin):
                xmin = boxes[idxs[idx]][0]
            if (boxes[idxs[idx]][1] < ymin):
                ymin = boxes[idxs[idx]][1]
            if (boxes[idxs[idx]][2] > xmax):
                xmax = boxes[idxs[idx]][2]
            if (boxes[idxs[idx]][3] > ymax):
                ymax = boxes[idxs[idx]][3]
        if (x1[i] - xmin > 0.1 * width):
            xmin = x1[i] - 0.1 * width
        if (y1[i] - ymin > 0.1 * height):
            ymin = y1[i] - 0.1 * height
        if (xmax - x2[i] > 0.1 * width):
            xmax = x2[i] + 0.1 * width
        if (ymax - y2[i] > 0.1 * height):
            ymax = y2[i] + 0.1 * height
        result_boxes.append([xmin, ymin, xmax, ymax, ave_prob / len(delete_idxs)])
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, delete_idxs)

    # return only the bounding boxes that were picked using the
    # integer data type
    # result = np.delete(boxes[pick],np.where(boxes[pick][:, 4] < 0.9)[0],  axis=0)
    # print boxes[pick]
    return result_boxes


def nms_max(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:, 4])

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # area of i.
        area_i = np.maximum(0, x2[i] - x1[i] + 1) * np.maximum(0, y2[i] - y1[i] + 1)
        area_array = np.zeros(len(idxs) - 1)
        area_array.fill(area_i)
        # compute the ratio of overlap
        overlap = (w * h) / (area[idxs[:last]] - w * h + area_array)
        # overlap = (w * h) / (area[idxs[:last]])
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    # result = np.delete(boxes[pick],np.where(boxes[pick][:, 4] < 0.9)[0],  axis=0)
    # print boxes[pick]
    return boxes[pick]


def convert_full_conv():
    # Load the original network and extract the fully connected layers' parameters.
    net = caffe.Net('deploy.prototxt',
                    'alexNet__iter_60000.caffemodel',
                    caffe.TEST)
    params = ['fc6', 'fc7', 'fc8_flickr']
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
    # Load the fully convolutional network to transplant the parameters.
    net_full_conv = caffe.Net('face_full_conv.prototxt',
                              'alexNet__iter_60000.caffemodel',
                              caffe.TEST)
    params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
    conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}
    for pr, pr_conv in zip(params, params_full_conv):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]
    net_full_conv.save('face_full_conv.caffemodel')


def face_detection(imgList):
    img_count = 0

    import numpy as np
    import cv2

    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # convert_full_conv()
    face_detection("lfw.txt")
