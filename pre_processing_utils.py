import os
import numpy as np
import tensorflow as tf
import cv2
import multiprocessing

from tqdm import tqdm
from matplotlib.image import imread
from multiprocessing import Pool
from PIL import Image


root_data_dir = './chest_xray'
test_path = root_data_dir + '/test'
train_path = root_data_dir + '/train'
validation_path = root_data_dir + '/val'
normal_path = '/NORMAL'
pneumonia_path = '/PNEUMONIA'

flip_transformation = 'flip'
rotation_transformation = 'rotation'


def check_data_exists():
    if not os.path.exists(root_data_dir) or not os.path.exists(test_path) or not os.path.exists(train_path) or not os.path.exists(validation_path):
        print('dataset has not been downloaded or is not within this directory')
    else:
        print('test data location = ' + test_path + "\ntraining data location = " +
              train_path + "\nvalidation data location = " + validation_path)


# returns the average dimensions of the images in the data-set
def get_average_dimensions():
    dim1 = []
    dim2 = []

    # Getting the dimensions of the testing data
    test_dim_normal_1, test_dim_normal_2 = get_dimensions_from_folder(
        test_path, normal_path)
    test_dim_pneumonia_1, test_dim_pneumonia_2 = get_dimensions_from_folder(
        test_path, pneumonia_path)

    dim1 += test_dim_normal_1 + test_dim_pneumonia_1
    dim2 += test_dim_normal_2 + test_dim_pneumonia_2

    # Getting the dimensions of the training data
    train_dim_normal_1, train_dim_normal_2 = get_dimensions_from_folder(
        train_path, normal_path)
    train_dim_pneumonia_1, train_dim_pneumonia_2 = get_dimensions_from_folder(
        train_path, pneumonia_path)

    dim1 += train_dim_normal_1 + train_dim_pneumonia_1
    dim2 += train_dim_normal_2 + train_dim_pneumonia_2

    # Getting the dimensions of the validation data
    val_dim_normal_1, val_dim_normal_2 = get_dimensions_from_folder(
        validation_path, normal_path)
    val_dim_pneumonia_1, val_dim_pneumonia_2 = get_dimensions_from_folder(
        validation_path, pneumonia_path)

    dim1 += val_dim_normal_1 + val_dim_pneumonia_1
    dim2 += val_dim_normal_2 + train_dim_pneumonia_2

    return [np.round(np.mean(dim1)), np.round(np.mean(dim2))]


def get_dimensions_from_folder(image_folder, path):
    dim1 = []
    dim2 = []

    for image_filename in os.listdir(image_folder + path):
        if (image_filename != '.DS_Store' and not is_transformation(image_filename)):
            img = imread(image_folder + path + '/' + image_filename)
            if (len(img.shape) > 2):
                d1, d2, _ = img.shape
            else:
                d1, d2 = img.shape
            dim1.append(d1)
            dim2.append(d2)
    return [dim1, dim2]


def flipImages():
    tasks = [
        [test_path, normal_path, flip_transformation],
        [test_path, pneumonia_path, flip_transformation],
        [train_path, normal_path, flip_transformation],
        [train_path, pneumonia_path, flip_transformation],
        [validation_path, normal_path, flip_transformation],
        [validation_path, pneumonia_path, flip_transformation]
    ]

    try:
        print("Performing flip transformations: ")
        process_pool = Pool()
        for _ in tqdm(process_pool.imap_unordered(apply_transformation_to_folder, tasks), desc='Data-Directories Transformed: ', total=len(tasks), unit=' DIR'):
            pass
        process_pool.close()
        process_pool.join()
    except ValueError:
        print("Critical error reached when applying transformations")


def rotateImages():
    tasks = [
        [test_path, normal_path, rotation_transformation],
        [test_path, pneumonia_path, rotation_transformation],
        [train_path, normal_path, rotation_transformation],
        [train_path, pneumonia_path, rotation_transformation],
        [validation_path, normal_path, rotation_transformation],
        [validation_path, pneumonia_path, rotation_transformation]
    ]
    try:
        print("Performing rotation transformations: ")
        process_pool = Pool()
        for _ in tqdm(process_pool.imap_unordered(apply_transformation_to_folder, tasks), desc='Data-Directories Transformed: ', total=len(tasks), unit=' DIR'):
            pass
        process_pool.close()
        process_pool.join()
    except ValueError:
        print("Critical error reached when applying transformations")


def apply_transformation_to_folder(task):
    image_folder = task[0]
    path = task[1]
    transformation = task[2]
    # Checking folder for transformation -- if transformation is already present we end execution before re-applying them
    for image_filename in os.listdir(image_folder + path):
        if (is_transformed_image(image_filename, transformation)):
            print(transformation + " has already been applied in " + image_folder)
            return
    # Applying specified transform
    for image_filename in os.listdir(image_folder + path):
        transformed_images = []
        if (image_filename != '.DS_Store' and not is_transformation(image_filename)):
            img = imread(image_folder + path + '/' + image_filename)

            # Converting grey-scale to RGB -- needed for tf.convert_to_tensor()
            if (len(img.shape) < 3):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            tf_img = tf.convert_to_tensor(img)
            image_filename = image_filename.split('.')[0]

            # Flip Transform
            if (transformation == flip_transformation):
                transformed_images += [[tf.image.flip_up_down(tf_img), image_filename + '-FlipUD'], [tf.image.random_flip_up_down(tf_img), image_filename + '-RandFlipUD'], [
                    tf.image.flip_left_right(tf_img), image_filename + '-FlipLR'], [tf.image.random_flip_left_right(tf_img), image_filename + '-RandFlipLR']]
            if (transformation == rotation_transformation):
                # k = number of anti-clockwise 90 degree rotations
                transformed_images += [[tf.image.rot90(tf_img, k=1), image_filename + '-Rotation90'], [
                    tf.image.rot90(tf_img, k=2), image_filename + '-Rotation180']]
            # Saving the specified transform to the file-system
            for transformed_image in transformed_images:
                img, imgName = transformed_image
                img_to_save = tf.io.encode_jpeg(img)
                tf.io.write_file(os.path.join(
                    image_folder+path + '/'+imgName+'.jpeg'), img_to_save)


def is_transformation(image_filename):
    return is_transformed_image(image_filename, flip_transformation) or is_transformed_image(image_filename, rotation_transformation)


def is_transformed_image(image_filename, transformation):
    name_with_extension = image_filename.split('/')[-1]
    image_name_lowercase = name_with_extension.split('.')[0].lower()
    return (transformation in image_name_lowercase)
