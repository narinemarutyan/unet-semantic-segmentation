import os

import cv2
import numpy as np
from PIL import Image
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
patch_size = 256


def load_images(root_directory):
    image_dataset = []
    for path, subdirs, files in os.walk(root_directory):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':
            images = os.listdir(path)
            for image_name in images:
                if image_name.endswith(".jpg"):
                    image = cv2.imread(os.path.join(path, image_name), 1)
                    SIZE_X = (image.shape[1] // patch_size) * patch_size
                    SIZE_Y = (image.shape[0] // patch_size) * patch_size
                    image = Image.fromarray(image)
                    image = image.crop((0, 0, SIZE_X, SIZE_Y))
                    image = np.array(image)
                    patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
                    print(f"Processing image: {image_name}, Size: {image.shape}")
                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            single_patch_img = patches_img[i, j, :, :]
                            single_patch_img = scaler.fit_transform(
                                single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(
                                single_patch_img.shape)
                            single_patch_img = single_patch_img[0]
                            image_dataset.append(single_patch_img)
                            if i == 0 and j == 0:  # Print the first patch of the first image for verification
                                print(f"First patch of {image_name}: {single_patch_img.shape}")
    print(np.array(image_dataset).shape)
    return np.array(image_dataset)


def load_masks(root_directory):
    mask_dataset = []
    for path, subdirs, files in os.walk(root_directory):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'masks':
            masks = os.listdir(path)
            for mask_name in masks:
                if mask_name.endswith(".png"):
                    mask = cv2.imread(os.path.join(path, mask_name), 1)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                    SIZE_X = (mask.shape[1] // patch_size) * patch_size
                    SIZE_Y = (mask.shape[0] // patch_size) * patch_size
                    mask = Image.fromarray(mask)
                    mask = mask.crop((0, 0, SIZE_X, SIZE_Y))
                    mask = np.array(mask)
                    patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)
                    print(f"Processing mask: {mask_name}, Size: {mask.shape}")
                    for i in range(patches_mask.shape[0]):
                        for j in range(patches_mask.shape[1]):
                            single_patch_mask = patches_mask[i, j, :, :]
                            single_patch_mask = single_patch_mask[0]
                            mask_dataset.append(single_patch_mask)
                            if i == 0 and j == 0:  # Print the first patch of the first mask for verification
                                print(f"First mask patch of {mask_name}: {single_patch_mask.shape}")
    return np.array(mask_dataset)
