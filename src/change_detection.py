import argparse

import cv2

from src.inference_segment import run_inference
from src.utils import load_model_for_inference


def detect_changes(image_path1: str, image_path2: str) -> cv2.Mat:
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    difference = cv2.absdiff(gray1, gray2)

    _, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    return thresh


def main(args):
    model = load_model_for_inference(args.model_path)
    change_mask = detect_changes(args.image_path1, args.image_path2)

    segmented_mask_image_path1 = run_inference(model=model, image_path=args.image_path1, patch_size=args.patch_size)
    segmented_mask_image_path2 = run_inference(model=model, image_path=args.image_path2, patch_size=args.patch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Change detection arguments')
    parser.add_argument("--model_path", type=str, required=True, help="Path to the segmentation model")
    parser.add_argument("--image_path1", type=str, required=True, help="Path to the 1st input image.")
    parser.add_argument("--image_path2", type=str, required=True, help="Path to the 2nd input image.")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch sizes")

    args = parser.parse_args()
    main(args)
