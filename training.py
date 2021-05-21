'''
You can use this file to train a new model. Make sure the dependencies are installed.
Install tensorflow: https://www.tensorflow.org/install
To start training, navigate in terminal to the directory where you will have two folders and this file.
To start training: $ python training.py
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflowjs as tfjs
import pathlib


# Passing path to data which will have individual folder of each category
data_dir = pathlib.Path('data')

# Use the following to count the number of data
# image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)

'''
We need to define some parameters at this point, regarding the images.
In our case, since the quantity of our data is small, it is better to have a
batch size that is smaller also. More information on the "batch size" can
be found online, and also how important is it for training.
The size of our images is around 200px x 200 px
'''

batch_size = 30
img_height = 120
img_width = 120


'''
Good practice in Machine Learning is to split the data to 80% for training and 20% for validatation.
This is one reason we need to have the same quantity of images for each class
'''

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_height),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# This can give us a preview of the classes
# class_names = train_ds.class_names
# print(class_names)


''' Setting for better performance '''

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE).shuffle(10)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# This will standardize values to 0 and 1, instead of the RGB values of 0-255

#normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
#normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

#normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

#resized_layer = layers.experimental.preprocessing.Resizing(img_height, img_width)
rescaled_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3))
#randomflip_layer = layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")
#randomrotation_layer = layers.experimental.preprocessing.RandomRotation(0.2)

normalized_ds = train_ds.map(lambda x, y: (rescaled_layer(x), y))
normalized_val = val_ds.map(lambda x, y: (rescaled_layer(x), y))
#randomflip_ds = resize_ds.map(lambda x, y: (randomflip_layer(x), y))
#randomrotation_ds = randomflip_ds.map(lambda x, y: (randomflip_layer(x),y))
#rescaled_ds = resize_ds.map(lambda x, y: (rescaled_layer(x), y))


# Creating the model; You must fine tune this model for high accurancy
# Adjust the num_classes depending on the number of category of images
num_classes = 18

model = Sequential([
# input_shape should be adjusted depending on the size of the images, the last digit is #regarding the color

  layers.Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 3)),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(128, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(128, (3, 3), activation='relu'),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])


# To view training and validation accuracy for each epoch

model.compile(optimizer=keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])


# Training; we set the number of epochs depending on the variery of images in a class

epochs=2
history = model.fit(
  normalized_ds,
  validation_data=normalized_val,
  epochs=epochs
)

# converting the model and then saving it in the folder called "pre_model"; make sure this 
# exist in the same directory

tfjs.converters.save_keras_model(model, "pre_models")

# Will display summary

model.summary()

# Visualizing

'''acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
'''
