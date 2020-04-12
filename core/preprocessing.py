from IPython.display import clear_output

import pre_processing_utils

pre_processing_utils.check_data_exists()


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
        pre_processing_utils.apply_transformation_to_folder(task, -1)
        work_done += 1
        update_progress(work_done / total_work, description)
    update_progress(1, description)


def equalize_data_with_rand_flips(root_path):
    normal_data_path = root_path + pre_processing_utils.normal_path
    normal_files = pre_processing_utils.get_num_files(normal_data_path)

    pneumonia_data_path = root_path + pre_processing_utils.pneumonia_path
    pneumonia_files = pre_processing_utils.get_num_files(pneumonia_data_path)

    diff = abs(normal_files - pneumonia_files)
    print(diff)
    if normal_files < pneumonia_files:
        task = [root_path, pre_processing_utils.normal_path,
                pre_processing_utils.equalize_data_transformation]
    else:
        task = [root_path, pre_processing_utils.pneumonia_path,
                pre_processing_utils.equalize_data_transformation]
    pre_processing_utils.apply_transformation_to_folder(task, diff)


def augment_training_data():
    root_data_path = pre_processing_utils.test_path
    # Defining the data paths for the given root dir
    normal_data_path = root_data_path + pre_processing_utils.normal_path
    pneumonia_data_path = root_data_path + pre_processing_utils.pneumonia_path
    # Equalizing the data before applying transformations
    if pre_processing_utils.get_num_files(normal_data_path) != \
            pre_processing_utils.get_num_files(pneumonia_data_path):
        equalize_data_with_rand_flips(root_data_path)
        perform_pre_processing_tasks(pre_processing_utils.get_flip_images_tasks(
            root_data_path), 'Flipping Images Progress:')
        perform_pre_processing_tasks(pre_processing_utils.get_rotate_images_tasks(
            root_data_path), 'Rotating Images Progress:')
        perform_pre_processing_tasks(pre_processing_utils.get_scale_images_tasks(
            root_data_path), 'Scaling Images Progress:')
        perform_pre_processing_tasks(pre_processing_utils.get_translation_image_tasks(
            root_data_path), 'Translating Images Progress')
        perform_pre_processing_tasks(pre_processing_utils.get_noise_image_tasks(
            root_data_path), 'Adding Noise to Images Progress')


augment_training_data()
