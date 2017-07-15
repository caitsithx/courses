from keras import applications, optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model

from constants import *


def get_dense_model(input_shape, dropout=0.5, batch_normalized=False):
    dense_model = Sequential()
    dense_model.add(Flatten(input_shape=input_shape))

    dense_model.add(Dense(256, activation='relu'))
    if batch_normalized:
        dense_model.add(BatchNormalization())
    dense_model.add(Dropout(dropout))

    dense_model.add(Dense(2, activation='softmax'))

    dense_model.summary()
    return dense_model


def get_compiled_dense_model(input_shape, dropout=0.5, batch_normalized=False):
    model = get_dense_model(input_shape, dropout, batch_normalized)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_pretrained_models():
    vgg16_model = applications.VGG16(weights='imagenet',
                                     include_top=False,
                                     input_shape=(img_rows, img_cols, img_channel))

    vgg19_model = applications.VGG19(include_top=False,
                                     weights='imagenet',
                                     input_shape=(img_rows, img_cols, img_channel))

    res_model = applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(img_channel, img_rows, img_cols))

    pretrained_models = {
        "vgg16": vgg16_model,
        "vgg19": vgg19_model,
        "resnet50": res_model
    }

    return pretrained_models


def get_compiled_pretrained_models():
    pretrained_models = get_pretrained_models()
    for k, model in pretrained_models.items():
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

    return pretrained_models


def get_models():
    pretrained_models = get_pretrained_models()
    models = {k: Model(inputs=m.input, outputs=get_dense_model(m.output_shape[1:])(m.output)) for k, m in
              pretrained_models.items()}

    for k, model in models.items():
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])
