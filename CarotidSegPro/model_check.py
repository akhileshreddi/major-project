# import numpy as np
# import tensorflow as tf
# import cv2
# import os

# IMAGE_SIZE = 128


# def read_image(path):
#     x = cv2.imread(path, cv2.IMREAD_COLOR)
#     x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#     x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
#     x = x / 255.0
#     return x


# def dice_coef(y_true, y_pred):
#     smooth = 1e-15
#     y_true = tf.keras.layers.Flatten()(y_true)
#     y_pred = tf.keras.layers.Flatten()(y_pred)
#     intersection = tf.reduce_sum(y_true * y_pred)
#     return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


# def dice_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)


# def create_model():
#     custom_objects = {'dice_loss': dice_loss, 'dice_coef': dice_coef}
#     options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
#     model_path = 'C:/Users/NAGARAJ K/Project-M/Flask_APP'
#     model_path = 'static/model/'
#     model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, options=options)
#     return model


# def predict(image_path):
#     model = create_model()
#     x = read_image(image_path)
#     x = np.expand_dims(x, axis=0)
#     preds = model.predict(x)
#     preds = (preds > 0.5).astype(np.uint8)
#     preds = preds[0] * 255
#     return preds

# import numpy as np
# import tensorflow as tf
# import cv2

# IMAGE_SIZE = 128


# def read_image(path):
#     x = cv2.imread(path, cv2.IMREAD_COLOR)
#     x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#     x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
#     x = x / 255.0
#     return x


# def dice_coef(y_true, y_pred):
#     smooth = 1e-15
#     y_true = tf.keras.layers.Flatten()(y_true)
#     y_pred = tf.keras.layers.Flatten()(y_pred)
#     intersection = tf.reduce_sum(y_true * y_pred)
#     return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


# def dice_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)


# # def create_model():
# #     custom_objects = {'dice_loss': dice_loss, 'dice_coef': dice_coef}
# #     model_path = 'Flask_App/static/model'
# #     model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost'))
# #     return model

# def create_model():
#     custom_objects = {'dice_loss': dice_loss, 'dice_coef': dice_coef}
#     model_path = 'Flask_App/static/model/'
#     model = tf.saved_model.load(model_path)
#     return model



# def predict(image_path):
#     model = create_model()
#     x = read_image(image_path)
#     x = np.expand_dims(x, axis=0)
#     preds = model.predict(x)
#     preds = (preds > 0.5).astype(np.uint8)
#     preds = preds[0] * 255
#     return preds



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


# def predict(image_path):
#     model = create_model()
#     x = read_image(image_path)
#     x = np.expand_dims(x, axis=0)
#     preds = model.predict(x)
#     preds = (preds > 0.5).astype(np.uint8)
#     preds = preds[0] * 255
#     return preds

def predict(image_path):
    model = create_model()
    x = read_image(image_path)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = (preds > 0.5).astype(np.uint8) * 255
    return preds[0]  # Return the predicted mask without the extra dimension

