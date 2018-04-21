import cv2
import glob
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D
from keras.models import Sequential, load_model
import numpy as np
import os

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

def load_images(image_file_names):
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

train_features, train_labels = load_images(TRAIN_IMAGES)
test_features, test_labels = load_images(TEST_IMAGES)

model = Sequential()
model.add(Conv2D(
    filters=16,
    kernel_size=7,
    strides=1,
    padding="same",
    activation="relu",
    input_shape=train_features.shape[1:]))
model.add(Conv2D(
    filters=12,
    kernel_size=5,
    strides=1,
    padding="same",
    activation="relu"))
model.add(Conv2D(
    filters=6,
    kernel_size=3,
    strides=1,
    padding="same",
    activation="relu"))
model.add(Conv2D(
    filters=1,
    kernel_size=1,
    strides=1,
    padding="same",
    activation="relu"))
model.compile(optimizer="adam", loss="mse")
model.summary()

model_checkpoint = ModelCheckpoint(
    "model.h5", monitor="val_loss", save_best_only=True, verbose=1)
early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=0.00001, patience=30)

model.fit(
    x=train_features,
    y=train_labels,
    validation_data=(test_features, test_labels),
    callbacks=[
        model_checkpoint,
        early_stopping
    ],
    epochs=1000)

model = load_model("model.h5")

for feature in np.concatenate([train_features, test_features]):
    feature = np.reshape(feature, (1,) + feature.shape)
    prediction = model.predict(feature)
    prediction = np.reshape(prediction, feature.shape[1:3])
    prediction = np.uint8(prediction * 255 / np.max(prediction))
    cv2.imshow("image", prediction)
    cv2.waitKey(0)
