import os

import cv2
import numpy as np
from PIL import Image
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
patch_size = 256


def preprocess(img_path, dataset):
    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    SIZE_X = (img.shape[1] // patch_size) * patch_size
    SIZE_Y = (img.shape[0] // patch_size) * patch_size
    img = Image.fromarray(img)
    img = img.crop((0, 0, SIZE_X, SIZE_Y))
    img = np.array(img)
    patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            single_patch = single_patch[0]
            dataset.append(single_patch)
    return dataset, (patches.shape[0], patches.shape[1])


def load_images(root_directory):
    dataset = []
    for path, sub_dirs, files in os.walk(root_directory):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':
            images = os.listdir(path)
            for image_name in images:
                if image_name.endswith(".jpg"):
                    image_path = os.path.join(path, image_name)
                    dataset, _ = preprocess(image_path=image_path,
                                            dataset=dataset)
    return np.array(dataset)


def load_masks(root_directory):
    dataset = []
    for path, sub_dirs, files in os.walk(root_directory):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'masks':
            masks = os.listdir(path)
            for mask_name in masks:
                if mask_name.endswith(".png"):
                    mask_path = os.path.join(path, mask_name)
                    dataset, _ = preprocess(img_path=mask_path,
                                            dataset=dataset)
    return np.array(dataset)
