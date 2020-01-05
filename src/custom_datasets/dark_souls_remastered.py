# example of progressively loading images from file
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# create a data generator
datagen = ImageDataGenerator()
# load and iterate training dataset

IMG_HEIGHT = 720//2
IMG_WIDTH = 1280//2
EPOCHS = 16
BATCH_SIZE = 32
train_image_generator = ImageDataGenerator(rescale=1./255)


alive_number = len(os.listdir("/source/custom_datasets/dark_souls/train/alive/"))
dead_number = len(os.listdir("/source/custom_datasets/dark_souls/train/dead/"))

total_number = alive_number + dead_number

train_data_gen = train_image_generator.flow_from_directory(
    directory=r"/source/custom_datasets/dark_souls/train/",
    target_size=(IMG_HEIGHT,IMG_WIDTH),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42
)
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch= total_number / BATCH_SIZE,
    epochs=EPOCHS,
)

#save the model
tf.saved_model.save(model, "/source/model/dark_souls_remastered/1")