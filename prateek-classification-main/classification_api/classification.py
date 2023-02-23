import pip
import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def create_model():
    """Configure Model Settings"""
    model = tf.keras.models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu',
                      input_shape=(128, 128, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3, 3), activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.0028)),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.75),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='softmax')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def preprocess(cat, split, label):
    """Splitting and Loading Images from a given Dataset"""
    train_images = []
    train_labels = []
    for i in os.listdir(cat):
        # Reading image as a pixel values matrix
        image = cv2.imread(cat + '/' + i)
        res = cv2.resize(image, dsize=(128, 128),
                         interpolation=cv2.INTER_CUBIC)  # Resize
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)  # Converting to grayscale
        train_images.append(gray)
        train_labels.append(label)

    size = len(train_images)
    return train_images[int(split*size):], train_images[:int(split*size)], train_labels[int(split*size):], train_labels[:int(split*size)]


def plot_model_stats(model, epochs):
    """Plot the statistics of the given model, and number of epochs applied"""

    acc = model.history['accuracy']
    val_acc = model.history['val_accuracy']

    loss = model.history['loss']
    val_loss = model.history['val_loss']

    epochs_range = range(epochs)

    # plotting accuracy and loss
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


def train_model(model_filename='my_model.h5', show_model_stats=False):
    """Preparing train and test sets"""
    train = []
    test = []
    labeltrain = []
    labeltest = []
    # Bed
    train_images, test_images, train_labels, test_labels = preprocess(
        'prateek-classification-main\Data for test\Bed', 0.3, 0)
    train.extend(train_images)
    test.extend(test_images)
    labeltrain.extend(train_labels)
    labeltest.extend(test_labels)

    # Chair
    train_images, test_images, train_labels, test_labels = preprocess(
        'prateek-classification-main\Data for test\Chair', 0.15, 1)
    train.extend(train_images)
    test.extend(test_images)
    labeltrain.extend(train_labels)
    labeltest.extend(test_labels)

    # Sofa
    train_images, test_images, train_labels, test_labels = preprocess(
        'prateek-classification-main\Data for test\Sofa', 0.15, 2)
    train.extend(train_images)
    test.extend(test_images)
    labeltrain.extend(train_labels)
    labeltest.extend(test_labels)

    # Converting to arrays
    train = np.array(train)
    test = np.array(test)
    labeltrain = np.array(labeltrain)
    labeltest = np.array(labeltest)
    train = train.reshape(train.shape[0], 128, 128, 1).astype('float32')
    train = (train - 127.5) / 127.5  # Normalize the images to [-1, 1]
    test = test.reshape(test.shape[0], 128, 128, 1).astype('float32')
    test = (test - 127.5) / 127.5  # Normalize the images to [-1, 1]

    model = create_model()

    # Summarize the Model (optional)
    # model.summary()

    # Perform the Fitting of the Model, 15 epochs
    epochs = 15
    history = model.fit(train, labeltrain, epochs=epochs,
                        validation_data=(test, labeltest), verbose=2)

    model.save(model_filename)

    # Optionally Display the statistics of the newly fitted model
    if show_model_stats:
        plot_model_stats(history)


def classify(image, model_filename='my_model.h5'):
    """Accepts an image of any size for OpenCV
    
    image must be in grayscale
    """

    loaded_model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), model_filename))


    # image to predict
    resized = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

    gray = np.array(resized)
    # reshaping
    gray = gray.reshape(1, 128, 128, 1).astype('float32')
    gray = (gray - 127.5) / 127.5
    # predicting
    pred = np.argmax(loaded_model.predict(gray))

    if pred == 0:
        return ("Bed")
    elif pred == 1:
        return ("Chair")
    elif pred == 2:
        return ("Sofa")
