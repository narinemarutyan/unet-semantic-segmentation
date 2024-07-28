# Semantic Segmentation with UNet
## Overview
This project involves developing a model to segment changes in two remote sensing images of the same geographic area. The goal is to input two images of identical resolution and output a multi-class mask that identifies different types of changes. These changes are classified into several categories: buildings, roads, vegetation, water, land, and areas that remain unlabeled.
## Classes
The model will classify changes into the following categories:
- **Building**: Changes involving buildings.
- **Land**: Changes related to the land surface without specific structures.
- **Road**: Changes related to road structures.
- **Vegetation**: Changes in areas covered by vegetation.
- **Water**: Changes involving bodies of water.
- **Unlabeled**: Areas where no specific changes can be classified.
## Setup
### Prerequisites
This project uses Poetry for package management to avoid dependency conflicts and maintain a clean working environment.
### Installation
If you don't have Poetry installed on your system, follow the installation instructions from [Poetry's official documentation](https://python-poetry.org/docs/).

To install the project dependencies, navigate to the project directory and run:
```bash
poetry install
```

## Training the Model
To start training the model, execute the following command:
```bash
python -m src.train --root_directory "datasets/aerial_image_segmentation" --save_path "models/segmentation_model.h5"
```

## Running segmentation inference on a single image
```bash
python -m src.inference_segment --model_path "models/unet_100_epochs.h5" --image_path "example_images/original_image.jpg"
```

## Running change detection inference
```bash
python -m src.change_detection --model_path "models/unet_100_epochs.h5" --image_path1 "example_images/changed_image.jpg" --image_path2 "example_images/original_image.jpg"
```

