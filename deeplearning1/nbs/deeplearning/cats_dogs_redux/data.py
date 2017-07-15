from keras.preprocessing.image import ImageDataGenerator
from constants import *


def get_data_batches(ppd_train_img_folder, ppd_valid_img_folder):
    # this is the augmentation configuration we will use for training
    train_data_gen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_data_gen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_batches = train_data_gen.flow_from_directory(
        ppd_train_img_folder,  # this is the target directory
        target_size=(TARGET_SIZE, TARGET_SIZE),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use categorical_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_batches = test_data_gen.flow_from_directory(
        ppd_valid_img_folder,
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=batch_size,
        class_mode='categorical')

    return train_batches, validation_batches
