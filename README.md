# OCT Retinal Layer Segmentation

This repository contains code for training and evaluating a UNet model for segmenting retinal layers in Optical Coherence Tomography (OCT) images.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Model Saving](#model-saving)
- [Acknowledgments](#acknowledgments)

## Overview

This project uses a UNet model to segment retinal layers from OCT images. The code includes data preprocessing, model training, evaluation, and visualization.

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/your-username/OCT-Retinal-Layer-Segmenter.git
cd OCT-Retinal-Layer-Segmenter
pip install -r requirements.txt
Ensure you have the following installed:

Python 3.x
Keras
TensorFlow
NumPy
OpenCV
Matplotlib
scikit-learn
Dataset
You can download the dataset from the following link:

Download OCT Retinal Layer Segmentation Dataset

Organize the dataset in the following structure:
OCT-Retinal-Layer-Segmenter/
│
├── X/      # Folder containing OCT images
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
│
└── Y/      # Folder containing corresponding masks
    ├── mask1.jpeg
    ├── mask2.jpeg
    └── ...

Update the TRAIN_PATH_X and TRAIN_PATH_Y variables in the code to point to your dataset directories.

Usage
To preprocess the data and start training the model, run the provided script:
python train.py
Data Preprocessing
The images and masks are resized to 640x640 pixels. The masks are also label-encoded to prepare them for training.
# Resizing images
SIZE_X = 640
SIZE_Y = 640
n_classes = 9  # Number of classes for segmentation

Model Training
The model is defined using the UNet architecture from simple_unet.py. The training process includes splitting the data into training and testing sets, normalizing the images, and one-hot encoding the masks.

model = multi_unet_model(n_classes=n_classes, IMG_HEIGHT=SIZE_Y, IMG_WIDTH=SIZE_X, IMG_CHANNELS=1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_cat, batch_size=16, epochs=2, validation_data=(X_test, y_test_cat), shuffle=False)

Evaluation
Evaluate the model on the test set and calculate the mean Intersection over Union (IoU) for the classes.
_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")

# Mean IoU
from keras.metrics import MeanIoU
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

Visualization
Plot the training and validation loss and accuracy, and visualize some predictions.
# Plot loss and accuracy
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot predictions
plt.subplot(232)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()

Model Saving
Save the trained model for future use.
model.save('retina_segmentation_8_layer_iter_3+20epochs.hdf5')
