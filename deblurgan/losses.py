import keras.backend as K
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

import numpy as np

# Note the image_shape must be multiple of patch_shape
image_shape = (256, 256, 3)

vgg = VGG16(weights='imagenet', include_top=False)
loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
loss_model.trainable = False

def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def perceptual_loss_100(y_true, y_pred):
    return 100 * perceptual_loss(y_true, y_pred)


def perceptual_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(loss_model(y_true) - loss_model(y_pred)))

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true*y_pred)


def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))

    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)

    return tf.reduce_mean(gradient_penalty)
