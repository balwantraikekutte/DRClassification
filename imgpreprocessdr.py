#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:11:23 2018

@author: ck807
"""

import cv2, glob, os
import numpy as np

input_path = "/local/data/chaitanya/DR/classification/trainingData/"
output_path = "/local/data/chaitanya/DR/classification/trainProcessed/"
train_files = glob.glob(os.path.join(input_path, "*.jpeg"))

scale = 256 # target radius

# helper functions

# estimate the area inside the image where the eye is present
# This is done to seperate the black border from the eye
def estimate_radius(img):
    mx = img[img.shape[0] // 2,:,:].sum(1)
    rx = (mx > mx.mean() / 10).sum() / 2
    
    my = img[:,img.shape[1] // 2,:].sum(1)
    ry = (my > my.mean() / 10).sum() / 2

    return (ry, rx)

# Crop the image using the region obtained from the first helper function
# this gives the images with most of the black border cropped out    
def crop_img(img, h, w):
    h_margin = (img.shape[0] - h) // 2 if img.shape[0] > h else 0
    w_margin = (img.shape[1] - w) // 2 if img.shape[1] > w else 0

    crop_img = img[h_margin:h + h_margin,w_margin:w + w_margin,:]

    return crop_img

# the outer circle is a dark one which can interfere with network learning something useful
def remove_outer_circle(a, p, r):
    b = np.zeros(a.shape, dtype=np.uint8)
    cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(r * p), (1, 1, 1), -1, 8, 0)
    
    return a * b + 128 * (1 - b)


for i in range(0, len(train_files)):
    img_file = train_files[i] # read file
    img = cv2.imread(img_file) # read image from file
    ry, rx = estimate_radius(img) # estimate radius of eye region in image
    resize_scale = scale / max(rx, ry)
    w = min(int(rx * resize_scale * 2), scale * 2)
    h = min(int(ry * resize_scale * 2), scale * 2)
    img_resize = cv2.resize(img.copy(), (0, 0), fx=resize_scale, fy=resize_scale) # resize image using radius
    img_crop = crop_img(img_resize.copy(), h, w) # crop image to remove black borders
    img_remove_outer = remove_outer_circle(img_crop.copy(), 0.9, scale)
    img_remove_outer = cv2.resize(img_remove_outer, (416, 416))
    base = os.path.basename(input_path + img_file)
    cv2.imwrite(output_path + base, img_remove_outer)

