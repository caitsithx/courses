DATA_DIR = '/Users/xiaoliangl/Documents/codes/deeplearning/data/cats-dogs-redux'
TRAIN_DIR = DATA_DIR + '/train'
TRAIN_PROCESSED_DIR = DATA_DIR + '/train_processed'
VALID_PROCESSED_DIR = DATA_DIR + '/valid_processed'
TEST_DIR = DATA_DIR + '/test'
TEST_PROCESSED_DIR = DATA_DIR + '/test_processed'
RESULT_DIR = DATA_DIR + '/result'

TARGET_SIZE = 224
img_rows, img_cols, img_channel = TARGET_SIZE, TARGET_SIZE, 3

batch_size = 50
nb_epochs = 90

num_of_valid_images = 2000
num_of_train_images = 22997

num_of_train_cats = 11497
num_of_train_dogs = 11500

num_of_valid_cats = 1000
num_of_valid_dogs = 1000

w_file=RESULT_DIR + '/cats-dogs-redux-{}-{}.h5'
