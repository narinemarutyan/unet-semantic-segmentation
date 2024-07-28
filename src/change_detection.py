import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.inference_segment import decode_segmentation_mask, reconstruct_patches
from src.preprocessing import preprocess
from src.utils import load_model_for_inference


def detect_changes(image_path1: str, image_path2: str) -> np.ndarray:
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    difference = cv2.absdiff(gray1, gray2)

    _, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    return thresh


def run_inference(model, image_path: str, patch_size: int) -> np.ndarray:
    preprocessed_image, original_shape = preprocess(img_path=image_path,
                                                    dataset=[],
                                                    patch_size=patch_size,
                                                    resizing="resize")
    preprocessed_image = np.array(preprocessed_image)
    preprocessed_image = preprocessed_image.reshape((-1, patch_size, patch_size, 3))
    predictions = model.predict(preprocessed_image)
    predicted_mask = np.argmax(predictions, axis=-1)
    full_mask = reconstruct_patches(patches=predicted_mask,
                                    patch_size=patch_size,
                                    original_shape=original_shape)
    return full_mask


def main(args):
    model = load_model_for_inference(args.model_path)

    change_mask = detect_changes(args.original_image, args.changed_image)

    segmented_mask_original_image = run_inference(model, args.original_image, args.patch_size)
    segmented_mask_original_image = cv2.resize(segmented_mask_original_image, (change_mask.shape[1], change_mask.shape[0]))
    segmented_mask_changed_image = run_inference(model, args.changed_image, args.patch_size)
    segmented_mask_changed_image = cv2.resize(segmented_mask_changed_image, (change_mask.shape[1], change_mask.shape[0]))


    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.imread(args.original_image))
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.imread(args.changed_image))
    plt.title('Changed image')

    plt.subplot(2, 2, 3)
    plt.imshow(change_mask, cmap='gray')
    plt.title('Detected Changes')

    plt.subplot(2, 2, 4)
    plt.imshow(decode_segmentation_mask(segmented_mask_original_image), alpha=0.5)
    plt.imshow(decode_segmentation_mask(segmented_mask_changed_image), alpha=0.5)
    plt.imshow(change_mask, cmap='gray', alpha=0.5)
    plt.title('Overlay of Changes and Segmentation')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Change detection and segmentation visualization')
    parser.add_argument("--model_path", type=str, required=True, help="Path to the segmentation model")
    parser.add_argument("--original_image", type=str, required=True, help="Path to the original image.")
    parser.add_argument("--changed_image", type=str, required=True, help="Path to the changed image.")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch sizes")

    args = parser.parse_args()
    main(args)