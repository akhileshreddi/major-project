import numpy as np
import tensorflow as tf
import cv2
import os
IMAGE_SIZE = 128


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x / 255.0
    return x


def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def create_model():
    model_path = 'static/model'
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def predict(image_path):
    model = create_model()
    x = read_image(image_path)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = (preds > 0.5).astype(np.uint8) * 255
    return preds[0]  # Return the predicted mask without the extra dimension

