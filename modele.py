import sys

import numpy as np

import keras

import cv2
import matplotlib.pyplot as plt
from PIL import Image

model_unet = keras.models.load_model("static/model/checkpoint_mini_unet.h5")

dir_path = "static/images/"
mask_path = "static/masks/"

def return_img(img_name):
    img = cv2.imread(dir_path+img_name)
    img = Image.fromarray(img.astype(np.uint8))
    return img

def cat(i):
    # void
    if i in {0, 1, 2, 3, 4, 5, 6}:
        return 0
    # flat
    elif i in {7, 8, 9, 10}:
        return 1
    # construction
    elif i in {11, 12, 13, 14, 15, 16}:
        return 2
    # object
    elif i in {17, 18, 19, 20}:
        return 3
    # nature
    elif i in {21, 22}:
        return 4
    # sky
    elif i in {23}:
        return 5
    # human
    elif i in {24, 25}:
        return 6
    # vehicle
    elif i in {26, 27, 28, 29, 30, 31, 32, 33, -1}:
        return 7

def return_mask(img_name):
    mask = cv2.imread(mask_path+img_name)
    res = np.asarray(np.empty((256,256,3)))

    for i in range(256):
        for j in range(256):
            pix = mask[i][j][0]
            for k in range(3):
                res[i][j][k] = cat(pix)
    
    return res

def treat_image(img_name):
    # Return ndarray of the mask
    img = cv2.imread(dir_path+img_name)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256))
    X_test = np.asarray(np.empty((1,256,256,3)))
    X_test[0] = img

    res = model_unet.predict(X_test, verbose=0)

    res_test = np.asarray(np.empty((256,256,3)))

    for i in range(256):
        for j in range(256):
            pix = np.argmax(res[0][i][j])
            for k in range(3):
                res_test[i][j][k] = pix

    return res_test

palette = {
    0:(0,0,0),
    4:(0,6*16+4,0),
    1:(0,0,8*16),
    3:(11*16,3*16,6*16),
    2:(255,0,0),
    7:(255,13*16+7,0),
    6:(0,255,0),
    5:(0,11*16+15,255)
}

def merge(mask, img_name):
    img = cv2.imread(dir_path+img_name)

    new_mask = np.asarray(np.empty((256,256,3)))

    for i in range(256):
        for j in range(256):
            for k in range(3):
                new_mask[i,j,k] = palette[mask[i,j,0]][k]

    res = cv2.addWeighted(img.astype(np.uint8), .7, new_mask.astype(np.uint8), .4, 0.0)

    res = Image.fromarray(res.astype(np.uint8))
    return res

def color_mask(mask):
    new_mask = np.asarray(np.empty((256,256,3)))

    for i in range(256):
        for j in range(256):
            for k in range(3):
                new_mask[i,j,k] = palette[mask[i,j,0]][k]

    res = Image.fromarray(new_mask.astype(np.uint8))
    return res