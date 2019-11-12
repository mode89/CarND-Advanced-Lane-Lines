import cv2
import glob
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D
from keras.models import Sequential, load_model
import numpy as np
import os
import tensorflow as tf

TRAIN_IMAGES = [
    "straight_lines1.jpg",
    "straight_lines2.jpg",
    "test2.jpg",
    "test3.jpg",
    "test4.jpg",
    "test5.jpg"
]

TEST_IMAGES = [
    "test1.jpg",
    "test6.jpg"
]

class Model:

    def __init__(self, topology=None):
        self.train_features, self.train_labels = self.load_images(TRAIN_IMAGES)
        self.test_features, self.test_labels = self.load_images(TEST_IMAGES)

        if topology is None:
            self.model = Sequential()
            self.model.add(Conv2D(
                filters=16,
                kernel_size=7,
                strides=1,
                padding="same",
                activation="relu",
                input_shape=self.train_features.shape[1:]))
            self.model.add(Conv2D(
                filters=12,
                kernel_size=5,
                strides=1,
                padding="same",
                activation="relu"))
            self.model.add(Conv2D(
                filters=6,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="relu"))
            self.model.add(Conv2D(
                filters=1,
                kernel_size=1,
                strides=1,
                padding="same",
                activation="relu"))
            self.model.compile(optimizer="adam", loss="mse")
            self.model.summary()
        else:
            firstLayer = topology[0]
            self.model = Sequential()
            self.model.add(Conv2D(
                filters=firstLayer["filters"],
                kernel_size=firstLayer["kernel_size"],
                strides=1,
                padding="same",
                activation="relu",
                input_shape=self.train_features.shape[1:]))
            for layer in topology[1:]:
                self.model.add(Conv2D(
                    filters=layer["filters"],
                    kernel_size=layer["kernel_size"],
                    strides=1,
                    padding="same",
                    activation="relu"))
            self.model.add(Conv2D(
                filters=1,
                kernel_size=1,
                strides=1,
                padding="same",
                activation="relu"))
            self.model.compile(optimizer="adam", loss="mse")
            self.model.summary()

    def load_images(self, image_file_names):
        feature_images = list()
        label_images = list()
        for file_name in image_file_names:
            feature_path = os.path.join("training_data/features", file_name)
            feature_image = cv2.imread(feature_path)
            feature_images.append(feature_image)
            label_path = os.path.join("training_data/labels", file_name)
            label_image = cv2.imread(label_path)
            label_images.append(label_image[:,:,0])
        features = np.float32(feature_images) / 255 - 0.5
        labels = np.float32(label_images) / 255
        labels = np.reshape(labels, labels.shape + (1,))
        return features, labels

    def train_model(self):
        model_checkpoint = ModelCheckpoint(
            "binary_filter.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=0)
        early_stopping = EarlyStopping(
            monitor="val_loss", min_delta=0.00001, patience=30)

        history = self.model.fit(
            x=self.train_features,
            y=self.train_labels,
            validation_data=(self.test_features, self.test_labels),
            callbacks=[
                early_stopping
            ],
            verbose=0,
            epochs=1000)
        valLossHistory = history.history["val_loss"]
        trainLossHistory = history.history["loss"]
        bestValLoss = min(valLossHistory)
        bestEpoch = valLossHistory.index(bestValLoss)
        bestTrainLoss = trainLossHistory[bestEpoch]
        print("Best validation loss: {}".format(bestValLoss))
        print("Best train loss: {}".format(bestTrainLoss))
        return bestValLoss, bestTrainLoss

    def load(self):
        self.model = load_model("binary_filter.h5")

    def process_image(self, image):
        feature = np.float32(image) / 255 - 0.5
        feature = np.reshape(feature, (1,) + image.shape)
        prediction = self.model.predict(feature)
        prediction = np.reshape(prediction, feature.shape[1:3])
        prediction = np.uint8(prediction * 255 / np.max(prediction))
        binary_image = np.zeros_like(prediction)
        binary_image[(prediction > 50)] = 255
        return binary_image

if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    set_session(session)

    binary_filter = Model()
    binary_filter.train_model()
    binary_filter.load()

    for file_name in TRAIN_IMAGES + TEST_IMAGES:
        file_path = os.path.join("training_data/features", file_name)
        image = cv2.imread(file_path)
        image = binary_filter.process_image(image)
        cv2.imshow("image", image)
        cv2.waitKey(0)
