import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from data import *
from models import *
from time import gmtime, strftime


def lr_schedule(epoch):
    if epoch <= 30:
        return 0.0001
    if epoch <= 60:
        return 0.00005
    else:
        return 0.00001


def train_models(train_img_folder, train_img_num, valid_img_folder, valid_img_num):
    pretrained_models = get_compiled_pretrained_models()
    train_generator, validation_generator = get_data_batches(train_img_folder, valid_img_folder)
    index = 0

    train_labels = np.array([0] * num_of_train_cats + [1] * num_of_train_dogs)
    valid_labels = np.array([0] * num_of_valid_cats + [1] * num_of_valid_dogs)

    dense_models = {}
    for key, model in pretrained_models:
        print ("train model: %d " % index)
        bottleneck_features_train = model.predict_generator(train_generator, train_img_num // batch_size)
        np.save(open(RESULT_DIR + '/bottleneck_features_train_%s.npy' % key, 'w'), bottleneck_features_train)
        bottleneck_features_validation = model.predict_generator(validation_generator, valid_img_num // batch_size)
        np.save(open(RESULT_DIR + '/bottleneck_features_valid_%s.npy' % key, 'w'), bottleneck_features_validation)

        dense_model = get_compiled_dense_model(model.output_shape[1:])
        dense_model.fit(bottleneck_features_train, train_labels,
                        epochs=nb_epochs,
                        batch_size=batch_size,
                        validation_data=(bottleneck_features_validation, valid_labels))

        dense_models[key] = dense_model

    return pretrained_models, dense_models


def train_final_models():
    return train_models(TRAIN_PROCESSED_DIR, num_of_train_images, VALID_PROCESSED_DIR, num_of_valid_images)


def train_sample_models():
    return train_models(TRAIN_PROCESSED_DIR, num_of_train_images, VALID_PROCESSED_DIR, num_of_valid_images)


def current_time():
    return strftime("%Y-%m-%d-%H:%M:%S", gmtime())
