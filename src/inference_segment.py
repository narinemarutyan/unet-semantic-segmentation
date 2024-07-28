from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

from src.preprocessing import preprocess
from src.utils import load_model_for_inference, class_colors


def reconstruct_patches(patches: np.ndarray, patch_size: int, original_shape: Tuple[int, int]):
    """
    Reconstruct the patches into the original image shape.

    Parameters
    ----------
    patches: np.ndarray
        Array of image patches.
    patch_size: int
        Size of each patch.
    original_shape: Tuple[int, int]
        Original shape of the image (height, width).

    Returns
    -------
    out: np.ndarray
    """
    full_recon = np.zeros((original_shape[0] * patch_size, original_shape[1] * patch_size))
    k = 0
    for i in range(0, full_recon.shape[0], patch_size):
        for j in range(0, full_recon.shape[1], patch_size):
            full_recon[i:i + patch_size, j:j + patch_size] = patches[k]
            k += 1
    return full_recon


def run_inference(model, image_path: str, patch_size: int):
    """
    Run inference on a single image using the loaded model.

    Parameters
    ----------
    model: Model
        The loaded segmentation model.
    image_path: str
        Path to the input image file.
    patch_size: int
        Size of the image patches.

    Returns
    -------
    out: np.ndarray
    """
    preprocessed_image, original_shape = preprocess(img_path=image_path,
                                                    dataset=[],
                                                    patch_size=patch_size)
    preprocessed_image = np.array(preprocessed_image)

    preprocessed_image = preprocessed_image.reshape((-1, patch_size, patch_size, 3))

    predictions = model.predict(preprocessed_image)
    predicted_mask = np.argmax(predictions, axis=-1)

    full_mask = reconstruct_patches(patches=predicted_mask,
                                    patch_size=patch_size,
                                    original_shape=original_shape)

    return full_mask


def decode_segmentation_mask(mask: np.ndarray) -> np.ndarray:
    """
    Decode the segmentation mask back to an RGB image using class colors.

    Parameters
    ----------
    mask: np.ndarray
        Segmentation mask where each pixel value corresponds to a class label.

    Returns
    -------
    out: np.ndarray
        Decoded RGB image where each class label is replaced with its corresponding color.
    """
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for c in range(len(class_colors)):
        rgb_mask[mask == c] = class_colors[c]
    return rgb_mask


def main(args):
    """
    Main function to run the inference and display the results.

    Parameters
    ----------
    args : Namespace
        Parsed command line arguments.
    """
    model = load_model_for_inference(args.model_path)

    predicted_mask = run_inference(model=model, image_path=args.image_path, patch_size=args.patch_size)

    decoded_mask = decode_segmentation_mask(predicted_mask)

    original_image = cv2.imread(args.image_path, 1)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(decoded_mask)
    plt.title('Predicted Mask')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation model inference arguments.')
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the image patches.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file.")

    args = parser.parse_args()
    main(args)
