import os
import numpy as np
import tensorflow as tf
import cv2

from matplotlib.image import imread
from IPython.display import clear_output

root_data_dir = os.path.abspath('./chest_xray')
test_path = root_data_dir + '/test'
train_path = root_data_dir + '/train'
validation_path = root_data_dir + '/val'
normal_path = '/NORMAL'
pneumonia_path = '/PNEUMONIA'

flip_transformation = 'flip'
rotation_transformation = 'rotation'
equalize_data_transformation = 'rand_flip_transform'
scale_transformation = 'scale'
translation_transformation = 'translate'
noise_transformation = 'noise'


def update_progress(progress, description):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait=True)
    text = description + \
        " [{0}] {1:.1f}%".format(
            "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


def perform_pre_processing_tasks(tasks, description):
    update_progress(0, description)
    work_done = 0
    total_work = len(tasks)
    for task in tasks:
        apply_transformation_to_folder(task, -1)
        work_done += 1
        update_progress(work_done / total_work, description)
    update_progress(1, description)


def equalize_data_with_rand_flips(root_path):
    normal_data_path = root_path + normal_path
    normal_files = get_num_files(normal_data_path)

    pneumonia_data_path = root_path + pneumonia_path
    pneumonia_files = get_num_files(pneumonia_data_path)

    diff = abs(normal_files - pneumonia_files)
    print(diff)
    if normal_files < pneumonia_files:
        task = [root_path, normal_path,
                equalize_data_transformation]
    else:
        task = [root_path, pneumonia_path,
                equalize_data_transformation]
    apply_transformation_to_folder(task, diff)


def augment_training_data():
    root_data_path = train_path
    # Defining the data paths for the given root dir
    normal_data_path = root_data_path + normal_path
    pneumonia_data_path = root_data_path + pneumonia_path
    # Equalizing the data before applying transformations
    if get_num_files(normal_data_path) != \
            get_num_files(pneumonia_data_path):
        equalize_data_with_rand_flips(root_data_path)
        perform_pre_processing_tasks(get_flip_images_tasks(
            root_data_path), 'Flipping Images Progress:')
        update_progress(1, 'Flipping Images Progress:')
        perform_pre_processing_tasks(get_rotate_images_tasks(
            root_data_path), 'Rotating Images Progress:')
        update_progress(1, 'Rotating Images Progress:')
        perform_pre_processing_tasks(get_scale_images_tasks(
            root_data_path), 'Scaling Images Progress:')
        update_progress(1, 'Scaling Images Progress:')
        perform_pre_processing_tasks(get_translation_image_tasks(
            root_data_path), 'Translating Images Progress:')
        update_progress(1, 'Translating Images Progress:')
        perform_pre_processing_tasks(get_noise_image_tasks(
            root_data_path), 'Adding Noise to Images Progress:')
        update_progress(1, 'Adding Noise to Images Progress:')


def get_num_files(folder):
    numFiles = 0
    for image_filename in os.listdir(folder):
        if image_filename.endswith('.jpeg'):
            numFiles += 1
    return numFiles


def check_data_exists():
    if not os.path.exists(root_data_dir) or \
            not os.path.exists(test_path) or \
            not os.path.exists(train_path) or \
            not os.path.exists(validation_path):
        print(
            'dataset is not within this directory -- please resolve before continuing....')
        print('dataset should be included at his notebook\'s level, with the following structure:')
        print('model.ipynb\squeezeNet.ipynb\ncore/\nchest_xray/\n\ttest/\n\\t\tNORMAL/\n\t\tPNEUMONIA/\n\ttrain/\n\t\tNORMAL/\n\t\tPNEUMONIA/\n\tval/\n\t\tNORMAL/\n\t\tPNEUMONIA/')
        raise AttributeError('Data not found')

    print('test data location = ' + test_path + "\ntraining data location = " +
          train_path + "\nvalidation data location = " + validation_path)


def get_flip_images_tasks(root_dir):
    tasks = [
        [root_dir, normal_path, flip_transformation],
        [root_dir, pneumonia_path, flip_transformation]
    ]
    return tasks


def get_rotate_images_tasks(root_dir):
    tasks = [
        [root_dir, normal_path, rotation_transformation],
        [root_dir, pneumonia_path, rotation_transformation],
    ]
    return tasks


def get_scale_images_tasks(root_dir):
    tasks = [
        [root_dir, normal_path, scale_transformation],
        [root_dir, pneumonia_path, scale_transformation]
    ]
    return tasks


def get_translation_image_tasks(root_dir):
    tasks = [
        [root_dir, normal_path, translation_transformation],
        [root_dir, pneumonia_path, translation_transformation]
    ]
    return tasks


def get_noise_image_tasks(root_dir):
    tasks = [
        [root_dir, normal_path, noise_transformation],
        [root_dir, pneumonia_path, noise_transformation]
    ]
    return tasks


def apply_transformation_to_folder(task, num_transformations=-1):
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
        if (image_filename.endswith('.jpeg') and not is_transformation(image_filename) and (num_transformations == -1 or num_transformations > 0)):
            img = imread(image_folder + path + '/' + image_filename)

            # Converting grey-scale to RGB -- needed for tf.convert_to_tensor()
            if (len(img.shape) < 3):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            tf_img = tf.convert_to_tensor(img)
            image_filename = image_filename.split('.')[0]

            # Flip Transform
            if (transformation == flip_transformation):
                transformed_images += [[tf.image.flip_up_down(tf_img), image_filename + '-FlipUD'], [
                    tf.image.flip_left_right(tf_img), image_filename + '-FlipLR']]
            if (transformation == equalize_data_transformation):
                transformation_type = np.random.randint(low=0, high=1)
                if (transformation_type == 0):
                    transformed_images += [[tf.image.random_flip_up_down(
                        tf_img), image_filename + '-equalize']]
                else:
                    transformed_images += [[tf.image.random_flip_left_right(
                        tf_img), image_filename + '-equalize']]
                num_transformations -= 1
            if (transformation == rotation_transformation):
                # k = number of anti-clockwise 90 degree rotations
                transformed_images += [
                    [tf.image.rot90(tf_img, k=2), image_filename + '-Rotation120']]
            if (transformation == scale_transformation):
                image_height = img.shape[0]
                image_width = img.shape[1]
                scale = 0.1
                # Tensor for up-scaled image dimensions
                scaled_dims_up_arr = np.array(
                    [image_height*scale + image_height, image_width*scale + image_width])
                scale_dims_up_tensor = tf.convert_to_tensor(
                    scaled_dims_up_arr, dtype=tf.int32)
                # Tensor for down-scaled image dimensions
                scaled_dims_down_arr = np.array(
                    [abs(image_height*scale - image_height), abs(image_width*scale - image_width)])
                scale_dims_down_tensor = tf.convert_to_tensor(
                    scaled_dims_down_arr, dtype=tf.int32)

                # Resizing the image by +10% -- tensor is of type float
                resized_image_up = tf.image.resize(
                    tf_img, size=scale_dims_up_tensor,
                    method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=True)
                resized_image_up_cropped = tf.image.crop_to_bounding_box(
                    resized_image_up, 0, 0, image_height, image_width)
                # Resizing the image by -10% -- tenfor is also of type float
                resized_image_down = tf.image.resize(
                    tf_img, size=scale_dims_down_tensor,
                    method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=True)
                resized_image_down_cropped = tf.image.crop_to_bounding_box(
                    resized_image_down, 0, 0, image_height, image_width)
                # Casting Tensor to type int so it can be saved to the file system
                resized_image_up_int = tf.cast(
                    resized_image_up_cropped, tf.uint8)
                resized_image_down_int = tf.cast(
                    resized_image_down_cropped, tf.uint8)

                # Cropping the image back to its original shape
                transformed_images += [[resized_image_up_int, image_filename+'-scaledUp'], [
                    resized_image_down_int, image_filename+'-scaledDown']]
            if transformation == translation_transformation:
                # Arrays and tensors defined for resizing later
                original_image_size = np.array(
                    [img.shape[0], img.shape[1]])
                original_image_size_tensor = tf.convert_to_tensor(
                    original_image_size, dtype=tf.int32)
                # Performing a random translation with 10% margin for width and height
                translated_image = tf.keras.preprocessing.image.random_shift(
                    tf_img, 0.1, 0.1)
                # Resize image to original size
                translated_image_resized = tf.image.resize(
                    translated_image, size=original_image_size_tensor, method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=True)
                translated_image_cropped_int = tf.cast(
                    translated_image_resized, tf.uint8)

                transformed_images += [[translated_image_cropped_int,
                                        image_filename+'-randomTranslate']]
            if transformation == noise_transformation:
                # Creating gaussian noise
                noise = tf.random.normal(shape=tf.shape(
                    tf_img), mean=0.0, stddev=1.0, dtype=tf.float32)
                noise_int = tf.cast(noise, tf.uint8)
                # Adding noise to image
                noisy_tensor = tf.add(tf_img, noise_int)
                transformed_images += [[noisy_tensor, image_filename+'-noise']]
            # Saving the specified transform to the file-system
            for transformed_image in transformed_images:
                img, imgName = transformed_image
                img_to_save = tf.io.encode_jpeg(img)
                tf.io.write_file(os.path.join(
                    image_folder+path + '/'+imgName+'.jpeg'), img_to_save)


def is_transformation(image_filename):
    return is_transformed_image(image_filename, flip_transformation) or \
        is_transformed_image(image_filename, rotation_transformation) or \
        is_transformed_image(image_filename, scale_transformation) or \
        is_transformed_image(image_filename, translation_transformation) or \
        is_transformed_image(image_filename, noise_transformation)


def is_transformed_image(image_filename, transformation):
    name_with_extension = image_filename.split('/')[-1]
    image_name_lowercase = name_with_extension.split('.')[0].lower()
    return transformation in image_name_lowercase


check_data_exists()
augment_training_data()
