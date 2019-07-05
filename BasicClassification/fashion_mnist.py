# Example of a basic artifical network (ANN) model being made, trained, and tested
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Loading the data sets which returns 4 Numpy arrays
# The images are 28x28 Numpy arrays, with pixels value ranging between 0 and 255
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Create label names since they are not included in the data set
# The Labels from the data set is given by an array of integers, ranging from 0 to 9
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#################################################

# Pre-processing the Data

# Plotting the image
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# We want to scale the values to a range of 0 to 1. Since each each image's pixel value is 0 to 255,
# in order to get it to range of 0 to 1, we divide all the image values by 255
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the first 25 images from the training set and display class name below image.
# plt.figure(figsize=(10, 10))                            # adjusting the window size
# for i in range(25):                                     # plotting 25 images
#     plt.subplot(5, 5, i+1)                              # creating subplots of 5 rows by 5 columns
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)     # plotting images
#     plt.xlabel(class_names[train_labels[i]])            # plotting labels underneath images
# plt.show()

#################################################
# Building the Model
# Building the model with layers using Keras library functions
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # flattens 2d array of 28x28 px into 1d array of 28*28=784
    keras.layers.Dense(128, activation=tf.nn.relu),  # has 128 nodes with relu activation
    keras.layers.Dense(10, activation=tf.nn.softmax)  # returns array of 10 prob scores that sum to 1
])

#################################################
# Compiling the the Model for training
model.compile(optimizer='adam',  # how the model is updated based on the data it sees and loss fn
              loss='sparse_categorical_crossentropy',  # measures how accurate the model is during training
              metrics=['accuracy'])  # monitor the training and testing steps

#################################################
# Training the Model
# Training the neural networks requires the following steps:
#     1. Feed the training data to the model, which in our case is the train_images and train_labels
#     2. The models learns to associate images and train_labels
#     3. We ask the model to make predictions about a test set, which in our case the test_images array. We
#         verify that the predictions match the labels from the test_labels array
model.fit(train_images, train_labels, epochs=5)

#################################################
# Evaluate Accuracy
# Compare how the model performs on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy:', test_acc)
# The acc on the test dataset is less than on the training set, this is an example of OVERFITTING. Overfitting is when
# a machine learning model performs worse on new data than on their training data.

#################################################
# Make Predictions
predictions = model.predict(test_images)  # array of 10 numbers, describes "confidence" of the model
print(predictions[0])  # higher the number, more confident of its decision
print('Prediction:', np.argmax(predictions[0]))  # confident that this image is an ankle boot or class_names[9]
print('Actual:', test_labels[0])  # print out what it actually is to confirm it is correct


# Graph to look at the full set of 10 class predictions
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Plotting individual images with their predictions of what it is
# i = 0p
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)
# plt.show()
#
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions,test_labels,test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,test_labels)
# plt.show()

# Plotting multiple images with their predictions
# num_rows = 5
# num_cols = 3
# num_images = num_cols * num_rows
# plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, predictions, test_labels, test_images)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, predictions, test_labels)
# plt.show()

# Grab an image from the test dataset and add to a batch where it is the only member
img = test_images[0]
img = (np.expand_dims(img, 0))
print(img.shape)

# model.predict returns a list of lists, one for each of image in the batch of data
predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)
