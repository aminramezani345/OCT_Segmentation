OCT Retinal Layer Segmentation
This repository contains code for training and evaluating a UNet model for segmenting retinal layers in Optical Coherence Tomography (OCT) images.

Table of Contents
Overview
Installation
Dataset
Usage
Data Preprocessing
Model Training
Evaluation
Visualization
Model Saving
Acknowledgments
Overview
This project uses a UNet model to segment retinal layers from OCT images. The code includes data preprocessing, model training, evaluation, and visualization.

Installation
To get started, clone this repository and install the required dependencies:

bash
Copy code
git clone https://github.com/your-username/OCT-Retinal-Layer-Segmenter.git
cd OCT-Retinal-Layer-Segmenter
pip install -r requirements.txt
Dataset
You can download the dataset from the following link:

Download OCT Retinal Layer Segmentation Dataset

Organize the dataset in the following structure:

graphql
Copy code
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
Data Preprocessing
The images and masks are resized to 640x640 pixels. The masks are also label-encoded to prepare them for training.

python
Copy code
# Resizing images
SIZE_X = 640
SIZE_Y = 640
n_classes = 9  # Number of classes for segmentation
Model Training
The model is defined using the UNet architecture from simple_unet.py. The training process includes splitting the data into training and testing sets, normalizing the images, and one-hot encoding the masks.

python
Copy code
model = multi_unet_model(n_classes=n_classes, IMG_HEIGHT=SIZE_Y, IMG_WIDTH=SIZE_X, IMG_CHANNELS=1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_cat, batch_size=16, epochs=2, validation_data=(X_test, y_test_cat), shuffle=False)
Evaluation
Evaluate the model on the test set and calculate the mean Intersection over Union (IoU) for the classes.

python
Copy code
_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")

# Mean IoU
from keras.metrics import MeanIoU
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())
Visualization
Plot the training and validation loss and accuracy, and visualize some predictions.

python
Copy code
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

python
Copy code
model.save('retina_segmentation_8_layer_iter_3+20epochs.hdf5')
Acknowledgments
This work was inspired by various open-source projects and research in the field of medical image analysis.
