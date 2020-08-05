# TensorFlow
import tensorflow as tf
import tensorflow_datasets as tfdata

tfdata.disable_progress_bar()

# Helper Libs
import math
import numpy as np
import matplotlib.pyplot as plt
import logging

# Fix bug:
# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Categories
category_names = ['T-shirt/top',
                  'Trouser',
                  'Pullover',
                  'Dress',
                  'Coat',
                  'Sandal',
                  'Shirt',
                  'Sneaker',
                  'Bag',
                  'Ankle boot']

def do_nomalization(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


def show_prediction(image, prediction, predict_label, true_label, pos):
    # image
    plt.subplot(pos[0], pos[1], pos[2])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, plt.cm.binary)
    # prediction
    plt.subplot(pos[0], pos[1], pos[2]+1)
    plt.ylim([0, 1])
    predict_bar = plt.bar(range(10), prediction, color='#777777')
    predict_bar[predict_label].set_color('red')
    predict_bar[true_label].set_color('blue')
    # label
    label_color = 'black'
    if predict_label == true_label:
        label_color = 'blue'
    else:
        label_color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format( category_names[predict_label],
                                          100 * prediction[predict_label],
                                          category_names[true_label]),
                                          color=label_color)


def main():
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    # Loading data from Fasion MNIST
    dataset, metadata = tfdata.load('fashion_mnist', as_supervised=True, with_info=True)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    num_train_dataset = metadata.splits['train'].num_examples
    num_test_dataset = metadata.splits['test'].num_examples

    # Do nomalization
    train_dataset = train_dataset.map(do_nomalization)
    test_dataset = test_dataset.map(do_nomalization)

    # Disply the first 25th train data
    i = 0
    for images, labels in train_dataset.take(25):
        images = images.numpy().reshape((28, 28))
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(category_names[labels])
        plt.imshow(images, cmap=plt.cm.binary)
        i += 1
    plt.show()

    # Build model
    in_layer = tf.keras.layers.Flatten(input_shape=(28, 28, 1))
    layer0 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
    out_layer = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
    model = tf.keras.Sequential([in_layer,
                                 layer0,
                                 out_layer])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy'])

    BATCH_SIZE = 32
    train_dataset = train_dataset.repeat().shuffle(num_train_dataset).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_test_dataset / BATCH_SIZE))
    test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_dataset / BATCH_SIZE))
    print("Test results: {}, {}".format(test_loss, test_accuracy))

    # Make prediction and explore
    for images, labels in test_dataset.take(1):
        images = images.numpy()
        labels = labels.numpy()
        predictions = model.predict(images)

    ITEMS = 6
    COL = ITEMS*2
    ROW = math.ceil(BATCH_SIZE/ITEMS)
    plt.figure(figsize=[COL*2,COL])
    for i in range(BATCH_SIZE):
        pos = [ROW, COL, 2*i+1]
        plt.subplot(ROW, COL, i*2+1)
        predict_label = np.argmax(predictions[i])
        true_label = labels[i]
        image = images[i].reshape((28, 28))
        show_prediction(image, predictions[i], predict_label, true_label, pos)
    plt.show()
