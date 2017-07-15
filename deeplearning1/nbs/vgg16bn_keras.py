from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras import applications, regularizers
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
    ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras.preprocessing import image

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape(
    (3, 1, 1))


def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1]  # reverse axis rgb->bgr


class Vgg16BnKeras:
    """The VGG 16 Imagenet model with Batch Normalization for the Dense Layers"""

    def __init__(self, output_size, img_size=(224, 224), weights_file_path=None):
        self.classes = []
        self.model = self.create(output_size, img_size)
        if weights_file_path is not None:
            self.load_weights(weights_file_path)

    def predict(self, imgs):
        all_preds = self.model.predict(imgs)
        idxs = np.argmax(all_preds, axis=1)
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes

    def create(self, output_size, size):
        if size != (224, 224):
            include_top = False

        vgg16 = applications.VGG16(weights='imagenet',
                                   include_top=include_top,
                                   input_shape=size + 3)

        dense_model = Vgg16BnKeras.create_dense_model(vgg16.output_shape[1:], output_size)

        return Model(Model(inputs=vgg16.input,
                           outputs=dense_model(vgg16.output)))

    def load_weights(self, weights_file_path):
        self.model.load_weights(weights_file_path)

    @staticmethod
    def create_dense_model(input_shape, output_size):
        dense_model = Sequential()
        dense_model.add(Flatten(input_shape=input_shape))

        fc_block_size = 1
        for i in range(fc_block_size):
            dense_model.add(Dense(256, activation='relu'))
            dense_model.add(BatchNormalization())
            dense_model.add(Dropout(0.5))  # 0.5

        dense_model.add(Dense(output_size, activation='softmax'))
        return dense_model

    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True,
                    batch_size=8, class_mode='categorical'):
        return gen.flow_from_directory(path, target_size=(224, 224),
                                       class_mode=class_mode, shuffle=shuffle,
                                       batch_size=batch_size)

    def finetune(self, batches):
        self.ft(batches.nb_class)

        classes = list(iter(batches.class_indices))
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes

    def compile(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=SGD(lr=1e-4, momentum=0.9),
                           metrics=['accuracy'])

    def fit_data(self, trn, labels, val, val_labels, nb_epoch=1, batch_size=64):
        self.model.fit(trn, labels, nb_epoch=nb_epoch,
                       validation_data=(val, val_labels), batch_size=batch_size)

    def fit(self, batches, val_batches, nb_epoch=1):
        self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample,
                                 nb_epoch=nb_epoch,
                                 validation_data=val_batches,
                                 nb_val_samples=val_batches.nb_sample)

    def test(self, path, batch_size=8):
        test_batches = self.get_batches(path, shuffle=False,
                                        batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches,
                                                          test_batches.nb_sample, verbose=1)
