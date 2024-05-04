from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
#import tensorflow.datasets as tfds
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
img_height, img_width = 224,224
batch_size = 20
class_names=["A","B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
train_ds = tf.keras.utils.image_dataset_from_directory(
    "C:\\Users\\Prem\\OneDrive\\Desktop\\archive (1)(2)\\asl_alphabet_train\\asl_alphabet_train",
    image_size = (img_height, img_width),
    batch_size = batch_size,
    color_mode="rgb"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "C:\\Users\\Prem\\OneDrive\\Desktop\\archive (1)(2)\\asl_alphabet_test\\asl_alphabet_test",
    image_size = (img_height, img_width),
    batch_size = batch_size,
    color_mode="rgb"
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "C:\\Users\\Prem\\OneDrive\\Desktop\\archive (1)(2)\\asl_alphabet_test\\asl_alphabet_test",
    image_size = (img_height, img_width),
    batch_size = batch_size,
    color_mode="rgb"
)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
    # flattening the convolutions
model.add(Flatten())
    # fully-connected layer
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(26, activation="softmax"))

model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)
#model.save("results/cifar10-model-v1.h5")
model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 15
)
model.save("15epochs.h5")
model.evaluate(test_ds)
for images,labels in test_ds.take(1):
    classifications=model(images)
    print(classifications)
    for i in range(len(images)):
        index=np.argmax(classifications[i])
        print("Pred:",class_names[index],"Real:",class_names[labels[i]])