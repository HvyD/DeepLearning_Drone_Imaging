
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hvyd
"""

# # Drone Image Segmentation
# 
# 

# In[ ]:


import os
import yaml
import datetime
import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
#from keras_fcn import FCN
#from voc_generator import PascalVocGenerator, ImageSetLoader
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN)
#from keras_fcn.callbacks import CheckNumericsOps


# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

global _SESSION
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
_SESSION = tf.Session(config=config)
K.set_session(_SESSION)

with open("init_args.yml", 'r') as stream:
    try:
        init_args = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# In[ ]:


checkpointer = ModelCheckpoint(
    filepath="/tmp/fcn_vgg16_weights.h5",
    verbose=1,
    save_best_only=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10, min_lr=1e-12)
early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=30)
nan_terminator = TerminateOnNaN()
csv_logger = CSVLogger('output/tmp_fcn_vgg16.csv')


# In[ ]:


datagen = PascalVocGenerator(image_shape=[224, 224, 3],
                                    image_resample=True,
                                    pixelwise_center=True,
                                    pixel_mean=[115.85100, 110.50989, 102.16182],
                                    pixelwise_std_normalization=True,
                                    pixel_std=[70.30930, 69.41244, 72.60676])


# In[ ]:


train_loader = ImageSetLoader(**init_args['image_set_loader']['train'])
val_loader = ImageSetLoader(**init_args['image_set_loader']['val'])


# In[ ]:


# model
fcn_vgg16 = FCN(input_shape=(224, 224, 3), classes=21, weight_decay=3e-3,
                weights='imagenet', trainable_encoder=True)


# In[ ]:


# optimizer
optimizer = keras.optimizers.Adam(1e-4)


# In[ ]:


fcn_vgg16.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


# In[ ]:


fcn_vgg16.fit_generator(
    datagen.flow_from_imageset(
        class_mode='categorical',
        classes=21,
        batch_size=1,
        shuffle=True,
        image_set_loader=train_loader),
    steps_per_epoch=1112,
    epochs=100,
    validation_data=datagen.flow_from_imageset(
        class_mode='categorical',
        classes=21,
        batch_size=1,
        shuffle=True,
        image_set_loader=val_loader),
    validation_steps=1111,
    verbose=1,
    callbacks=[lr_reducer, early_stopper, csv_logger, checkpointer, nan_terminator])


# In[ ]:


fcn_vgg16.save('output/fcn_vgg16.h5')


# In[ ]:


"""Fully Convolutional Neural Networks."""
from __future__ import (
    absolute_import,
    unicode_literals
)
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Activation, Reshape


from keras_fcn.encoders import VGG16, VGG19
from keras_fcn.decoders import VGGDecoder, VGGUpsampler


# In[ ]:


# Model detailed
def FCN_VGG16(input_shape, classes, weight_decay=0.,
              trainable_encoder=True, weights=None):
    """Fully Convolutional Networks for semantic segmentation with VGG16.
    # Arguments
        input_shape: input image shape
        classes: number of classes
        trainable_encoder: Bool whether the weights of encoder are trainable
        weights: pre-trained weights to load (None for training from scratch)
    # Returns
        A Keras model instance
    """
    # input
    inputs = Input(shape=input_shape)

    # Get the feature pyramid [drop7, pool4, pool3] from the VGG16 encoder
    pyramid_layers = 3
    encoder = VGG16(inputs, weight_decay=weight_decay,
                    weights=weights, trainable=trainable_encoder)
    feat_pyramid = encoder.outputs[:pyramid_layers]

    # Append image to the end of feature pyramid
    feat_pyramid.append(inputs)

    # Decode feature pyramid
    outputs = VGGUpsampler(feat_pyramid, scales=[1, 1e-2, 1e-4], classes=classes, weight_decay=weight_decay)

    # Activation TODO{jihong} work only for channels_last
    scores = Activation('softmax')(outputs)

    # return model
    return Model(inputs=inputs, outputs=scores)


# In[ ]:


def forward_propagation(...):
    # from X to cost


# In[ ]:


def mean_categorical_crossentropy(y_true, y_pred):
    if K.image_data_format() == 'channels_last':
        loss = K.mean(keras.losses.categorical_crossentropy(y_true, y_pred), axis=[1, 2])
    elif K.image_data_format() == 'channels_first':
        loss = K.mean(keras.losses.categorical_crossentropy(y_true, y_pred), axis=[2, 3])
    return loss

def flatten_categorical_crossentropy(classes):
    def f(y_true, y_pred):
        y_true = K.reshape(y_true, (-1, classes))
        y_pred = K.reshape(y_pred, (-1, classes))
        return keras.losses.categorical_crossentropy(y_true, y_pred)
    return f

