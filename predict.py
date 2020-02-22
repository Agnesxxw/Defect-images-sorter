# coding: utf-8

# import the necessary packages

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
# import  matplotlib
# matplotlib.use("Agg")
# import numpy as np
# import argparse
import random
import cv2
import os
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import regularizers


# coding=utf-8
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import add, Flatten
# from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.layers import Conv2D,MaxPool2D
import shutil


def load_data(path):
    print(path)

    labels_list=os.listdir(path);
    labels_list.sort();
    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    # print(imagePaths)
    # dda
    random.seed(42)
    random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        # print(imagePath)
        # print(imagePath)
        image = cv2.imread(imagePath)
        image=image[:,:,0];
        # print(image.shape)
        image = cv2.resize(image, (norm_size, norm_size))

        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        # print(imagePath.split(os.path.sep)[-2])
        label = labels_list.index(imagePath.split(os.path.sep)[-2])
        labels.append(label)

        ########
        # image = image / 255.0
        # image = np.expand_dims(image, axis=0)
        # result = model.predict(image)[0]
        # proba = np.max(result)
        # print(np.where(result == proba))
        # number = np.where(result == proba)[0]
        # print(number)
        # if number==label:
        #     if not os.path.exists(os.path.join("selectImages",imagePath.split(os.path.sep)[-2])):
        #         os.mkdir(os.path.join("selectImages",imagePath.split(os.path.sep)[-2]))
        #     shutil.copy(imagePath,os.path.join("selectImages",imagePath.split(os.path.sep)[-2],imagePath.split(os.path.sep)[-1]))
        #
        # print(labels_list[number[0]],imagePath.split(os.path.sep)[-2],label)
        ##########




    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)
    return data, labels




CLASS_NUM=0
EPOCHS = 100
INIT_LR = 1e-4
BS = 32
norm_size = 200
depth=1
model=keras.models.load_model("best.hdf5")
if __name__ == '__main__':
    path="./train"
    labels_list = os.listdir(path);
    labels_list.sort();
    imagePaths = sorted(list(paths.list_images(path)))
    imagePath=".\\train\\chipping\\_50026950203161902BW000157_X1_Y11_CAM20_20190225_104042.106.jpg"

    image = cv2.imread(imagePath)
    image = image[:, :, 0];
    # print(image.shape)
    image = cv2.resize(image, (norm_size, norm_size))

    image = img_to_array(image)
    print(imagePath.split(os.path.sep))
    label = labels_list.index(imagePath.split(os.path.sep)[-2])

    #######
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    result = model.predict(image)[0]
    proba = np.max(result)
    print(np.where(result == proba))
    number = np.where(result == proba)[0]


    print("This is {} {}%".format(labels_list[number[0]],proba*100))





