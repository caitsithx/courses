import multiprocessing
from glob import glob
from os import path, mkdir

from concurrent.futures import ProcessPoolExecutor

from constants import *
from deeplearning.image_utils import preprocess_image, check_mkdir


def create_class_dir(class_name):
    class_dir = TRAIN_PROCESSED_DIR + "/" + class_name
    if not path.exists(class_dir):
        mkdir(class_dir)


def classify_train_images():
    image_paths = glob(TRAIN_DIR + "/*.jpg")

    classified_count = 0
    for image_path in image_paths:
        path_parts = image_path.split("/")
        image_name = path_parts[len(path_parts) - 1]
        class_name = image_name.split(".")[0]
        image_id = image_name.split(".")[1]
        target_path = TRAIN_PROCESSED_DIR + "/" + class_name + "/" + image_id + ".jpg"
        create_class_dir(class_name)
        preprocess_image(image_path, target_path, TARGET_SIZE)
        if classified_count % 100 == 0:
            print "classified %d images" % classified_count


def classify_train_image(image_path, lock):
    lock.acquire()
    path_parts = image_path.split("/")
    image_name = path_parts[len(path_parts) - 1]
    class_name = image_name.split(".")[0]
    image_id = image_name.split(".")[1]
    target_path = TRAIN_PROCESSED_DIR + "/" + class_name + "/" + image_id + ".jpg"
    create_class_dir(class_name)
    preprocess_image(image_path, target_path, TARGET_SIZE)
    lock.release()


def classify_train_images_concurrent():
    l = multiprocessing.Lock()
    image_paths = glob(TRAIN_DIR + "/*.jpg")

    image_count = len(image_paths)
    with ProcessPoolExecutor(max_workers=4) as executor:
        for image_path in image_paths:
            executor.submit(classify_train_image, image_path, l)


if __name__ == '__main__':
    check_mkdir(TRAIN_PROCESSED_DIR)
    classify_train_images()
