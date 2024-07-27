import warnings

from src.preprocessing import load_images, load_masks

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from src.model import multi_unet_model, jacard_coef
from .utils import class_colors, rgb_to_2D_label, dice_loss_plus_1focal_loss, load_model_for_inference

import os

os.environ["SM_FRAMEWORK"] = "tf.keras"


def train_model(X_train, y_train, X_test, y_test, n_classes=6):
    model = multi_unet_model(n_classes=n_classes, img_height=256, img_width=256, img_channels=3)
    metrics = ['accuracy', jacard_coef]
    total_loss = dice_loss_plus_1focal_loss()
    model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
    model.summary()
    history = model.fit(X_train, y_train,
                        batch_size=16,
                        verbose=1,
                        epochs=1,
                        validation_data=(X_test, y_test),
                        shuffle=False)
    return model, history


def main():
    print("Starting the segmentation model training...")

    root_directory = 'datasets/aerial_image_segmentation/'
    print(f"Root directory set to {root_directory}")

    image_dataset = load_images(root_directory)
    print("Images loaded successfully.")
    mask_dataset = load_masks(root_directory)
    print("Masks loaded successfully.")

    labels = []
    for mask in mask_dataset:
        label = rgb_to_2D_label(mask, class_colors)
        labels.append(label)
    print("RGB masks converted to 2D labels.")

    labels = np.array(labels)
    labels = np.expand_dims(labels, axis=3)

    n_classes = len(np.unique(labels))
    labels_cat = to_categorical(labels, num_classes=n_classes)
    print(f"Total number of classes: {n_classes}")

    X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20, random_state=42)
    print("Data split into training and testing sets.")

    model, history = train_model(X_train, y_train, X_test, y_test, n_classes)
    print("Model training completed.")

    save_model_path = 'models/segmentation_model.h5'
    model.save(save_model_path)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['jacard_coef']
    val_acc = history.history['val_jacard_coef']

    plt.plot(epochs, acc, 'y', label='Training IoU')
    plt.plot(epochs, val_acc, 'r', label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.show()

    loaded_model = load_model_for_inference(save_model_path)

    test_image = X_test[0]
    prediction = loaded_model.predict(np.expand_dims(test_image, axis=0))
    predicted_mask = np.argmax(prediction, axis=-1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title('Test Image')
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask[0], cmap='jet')
    plt.title('Predicted Mask')
    plt.show()


if __name__ == "__main__":
    main()
    loaded_model = load_model_for_inference('models/segmentation_model.h5')
