import argparse
import os
import warnings

import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard

from src.metrics import jaccard_coef
from src.model import multi_unet_model
from src.preprocessing import load_images, load_masks
from .utils import class_colors, rgb_to_2D_label, dice_loss_plus_1focal_loss

os.environ["SM_FRAMEWORK"] = "tf.keras"
warnings.filterwarnings("ignore")


def main(args: argparse.Namespace) -> None:
    """
    Main function to train the segmentation model.

    Parameters
    ----------
    args: argparse.Namespace
        Command line arguments for training function.

    Returns
    -------
    out : None
    """
    image_dataset = load_images(root_directory=args.root_directory, patch_size=args.patch_size)
    mask_dataset = load_masks(root_directory=args.root_directory, patch_size=args.patch_size)

    labels = []
    for mask in mask_dataset:
        label = rgb_to_2D_label(mask, class_colors)
        labels.append(label)

    labels = np.array(labels)
    labels = np.expand_dims(labels, axis=3)

    labels_cat = to_categorical(labels, num_classes=len(np.unique(labels)))

    X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20)

    model = multi_unet_model(n_classes=args.n_classes, img_height=args.patch_size, img_width=args.patch_size,
                             img_channels=3)
    metrics = ['accuracy', jaccard_coef]
    total_loss = dice_loss_plus_1focal_loss()
    model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
    model.summary()

    tensorboard_callback = TensorBoard(log_dir=args.log_dir, histogram_freq=1)

    model.fit(x=X_train, y=y_train,
              batch_size=args.batch_size,
              verbose=1,
              epochs=args.epochs,
              validation_data=(X_test, y_test),
              shuffle=False,
              callbacks=[tensorboard_callback])

    save_model_path = args.save_path
    model.save(save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation model training arguments.')
    parser.add_argument('--root_directory', type=str, help='Root directory of the dataset.')
    parser.add_argument('--n_classes', type=int, default=6, help='Number of classes.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size for data preprocessing.')
    parser.add_argument('--save_path', type=str, help='Path to save the trained model.')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save TensorBoard logs.')

    args = parser.parse_args()
    main(args)
