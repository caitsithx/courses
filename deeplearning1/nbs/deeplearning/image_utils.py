from glob import glob

from os import mkdir, path
import numpy as np
from skimage import io, transform


def crop_from_center(img, width, height):
    y = img.shape[0]
    x = img.shape[1]
    offset_x = x // 2 - (width // 2)
    offset_y = y // 2 - (height // 2)
    return img[offset_y:offset_y + height, offset_x:offset_x + width]


def transform_image(file_path, target_size):
    # image will be resized to a square, so width and height is both target_size.
    # read image
    img = io.imread(file_path)
    # plt.imshow(img)

    if img.shape[0] > img.shape[1]:
        tile_size = (
        int(img.shape[0] * target_size / img.shape[1]), target_size)
    else:
        tile_size = (
        target_size, int(img.shape[1] * target_size / img.shape[0]))
    resized_image = transform.resize(img, tile_size,
                                     preserve_range=True)  # ranged from 0 ~ 255, float64

    cropped_image = crop_from_center(resized_image, target_size, target_size)
    # offset = [(tile_size[0] - target_size) // 2, (tile_size[1] - target_size) // 2]
    # cropped_image = resized_image[offset[0]:(offset[0] + target_size), offset[1]:(offset[1] + target_size)]

    # print cropped_image.shape
    # print cropped_image.dtype
    # print cropped_image.nbytes out of memory, almost 1M per image
    return cropped_image.astype(np.uint8)


def preprocess_image(img_path, target_path, target_size):
    io.imsave(target_path, transform_image(img_path, target_size))


def check_mkdir(dir_path):
    if not path.exists(dir_path):
        mkdir(dir_path)


def preprocess_images(img_folder, ppd_img_folder):
    image_paths = glob(img_folder + "/*.jpg")
    image_paths = np.random.permutation(image_paths)
    i = 0
    for img_path in image_paths:
        parts = img_path.split("/")
        file_name = parts[len(parts) - 1]
        io.imsave(ppd_img_folder + "/" + file_name, transform_image(img_path))
        i += 1
        if i % 100 == 0:
            print "pre-processed %d images" % i
