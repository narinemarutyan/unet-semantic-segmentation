import os

import cv2
import numpy as np
from PIL import Image
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


def preprocess(img_path: str, dataset: list[np.ndarray], patch_size: int):
    """
    Preprocesses an image by reading, converting to RGB, resizing, and splitting it into patches.

    Parameters
    ----------
    img_path: str
        Path to the image file.
    dataset: list of np.ndarray
        List to append the image patches to.
    patch_size: int
        Size of the patches to split the image into.

    Returns
    -------
    out: tuple
    """
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


def load_images(root_directory: str, patch_size: int):
    """
    Loads and preprocesses all images in the 'images' directory of the root directory.

    Parameters
    ----------
    root_directory: str
        Path to the root directory containing images.
    patch_size: int
        Size of the patches to split the images into.

    Returns
    -------
    out: np.ndarray
    """
    dataset = []
    for path, sub_dirs, files in os.walk(root_directory):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':
            images = os.listdir(path)
            for image_name in images:
                if image_name.endswith(".jpg"):
                    image_path = os.path.join(path, image_name)
                    dataset, _ = preprocess(img_path=image_path,
                                            dataset=dataset,
                                            patch_size=patch_size)
    return np.array(dataset)


def load_masks(root_directory: str, patch_size: int):
    """
    Loads and preprocesses all masks in the 'masks' directory of the root directory.

    Parameters
    ----------
    root_directory: str
        Path to the root directory containing masks.
    patch_size: int
        Size of the patches to split the masks into.

    Returns
    -------
    out: np.ndarray
    """
    dataset = []
    for path, sub_dirs, files in os.walk(root_directory):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'masks':
            masks = os.listdir(path)
            for mask_name in masks:
                if mask_name.endswith(".png"):
                    mask_path = os.path.join(path, mask_name)
                    dataset, _ = preprocess(img_path=mask_path,
                                            dataset=dataset,
                                            patch_size=patch_size)
    return np.array(dataset)
