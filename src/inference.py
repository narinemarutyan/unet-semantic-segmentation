import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import patch_size, preprocess
from src.utils import load_model_for_inference, class_colors


def reconstruct_patches(patches, original_shape):
    """
    Reconstruct the patches into the original image shape.
    """
    full_recon = np.zeros((original_shape[0] * patch_size, original_shape[1] * patch_size))
    k = 0
    for i in range(0, full_recon.shape[0], patch_size):
        for j in range(0, full_recon.shape[1], patch_size):
            full_recon[i:i + patch_size, j:j + patch_size] = patches[k]
            k += 1
    return full_recon


def run_inference(model, image_path):
    """
    Run inference on a single image using the loaded model.
    """
    preprocessed_image, original_shape = preprocess(img_path=image_path, dataset=[])
    preprocessed_image = np.array(preprocessed_image)

    preprocessed_image = preprocessed_image.reshape((-1, patch_size, patch_size, 3))

    predictions = model.predict(preprocessed_image)
    predicted_mask = np.argmax(predictions, axis=-1)

    full_mask = reconstruct_patches(predicted_mask, original_shape)

    return full_mask


def decode_segmentation_mask(mask):
    """
    Decode the segmentation mask back to an RGB image using class colors.
    """
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for c in range(len(class_colors)):
        rgb_mask[mask == c] = class_colors[c]
    return rgb_mask


def main():
    model_path = 'models/segmentation_model.h5'
    model = load_model_for_inference(model_path)

    image_path = 'datasets/aerial_image_segmentation/Tile 1/images/image_part_001.jpg'
    predicted_mask = run_inference(model, image_path)

    decoded_mask = decode_segmentation_mask(predicted_mask)

    original_image = cv2.imread(image_path, 1)
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
    main()
