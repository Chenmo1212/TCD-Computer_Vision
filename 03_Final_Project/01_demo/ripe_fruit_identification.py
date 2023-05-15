import cv2
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


def get_bboxes(filename):
    # load the bounding box data
    bbox_data = np.loadtxt(filename, delimiter=' ')

    # extract the bounding boxes from the data
    bbox = []
    for row in bbox_data:
        ripeness_class_id = int(row[0])
        center_x = row[1] * ori.shape[1]
        center_y = row[2] * ori.shape[0]
        width_x = row[3] * ori.shape[1]
        width_y = row[4] * ori.shape[0]

        # create the bounding box
        xmin = round((center_x - width_x / 2))
        xmax = round((center_x + width_x / 2))
        ymin = round((center_y - width_y / 2))
        ymax = round((center_y + width_y / 2))
        bbox.append([xmin, ymin, xmax, ymax, ripeness_class_id])
    return bbox


def get_train_data(box, image):
    # create the input and output data for the model
    x_ = []
    y_ = []
    for bbox in box:
        # extract the features from the image
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        instance_img = image[ymin:ymax, xmin:xmax]

        instance_img = cv2.resize(instance_img, (256, 256))

        # add the features and labels to the input and output data
        x_.append(instance_img)
        y_.append(bbox[4])

    # convert the input and output data to numpy arrays
    x_ = np.array(x_)
    y_ = np.array(y_)

    return x_, y_


def train_model(model, x_train, y_train):
    # create a deep learning model
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # compile and fit the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)


if __name__ == '__main__':
    # load the original image
    ori = cv2.imread('8_ori.png')
    # load the instance+ripeness segmentation image
    instance_ripe_seg = cv2.imread('ripeness.png', cv2.IMREAD_COLOR)

    bboxes = get_bboxes("8.txt")
    # print(bboxes)

    X, y = get_train_data(bboxes, instance_ripe_seg)

    print(X.shape, len(X))

    model = Sequential()

    # Convert NumPy array to Tensor
    # from keras.preprocessing.image import array_to_tensor
    # X = array_to_tensor(X, data_format='channels_last')
    train_model(model, X, y)
