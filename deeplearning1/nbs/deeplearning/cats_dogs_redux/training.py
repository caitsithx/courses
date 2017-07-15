from time import gmtime, strftime

from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from data import *
from models import *


def lr_schedule(epoch):
    if epoch <= 30:
        return 0.0001
    if epoch <= 60:
        return 0.00005
    else:
        return 0.00001


def train_models(train_img_folder, train_img_num, valid_img_folder, valid_img_num):
    models = get_models()
    train_generator, validation_generator = get_data_batches(train_img_folder, valid_img_folder)
    index = 0
    for key, model in models:
        print ("train model: %d " % index)
        model.fit_generator(
            train_generator,
            steps_per_epoch=train_img_num // batch_size,
            epochs=nb_epochs,
            callbacks=[ModelCheckpoint(w_file.format(key, current_time()),
                                       monitor='val_acc',
                                       save_best_only=True),
                       LearningRateScheduler(lr_schedule)],
            validation_data=validation_generator,
            validation_steps=valid_img_num // batch_size)

    return models


def train_final_models():
    return train_models(TRAIN_PROCESSED_DIR, num_of_train_images, VALID_PROCESSED_DIR, num_of_valid_images)


def train_sample_models():
    return train_models(TRAIN_PROCESSED_DIR, num_of_train_images, VALID_PROCESSED_DIR, num_of_valid_images)


def current_time():
    return strftime("%Y-%m-%d-%H:%M:%S", gmtime())
