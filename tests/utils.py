# %% Packages

import os
import pickle
from pyhocon import ConfigTree

# %% Functions


def load_pickle(loading_path: str):
    """This method loads the file at the specified path

    :param loading_path: Path at which object is saved
    :type loading_path: str
    :return: Desired file
    :rtype: Could be basically anything
    """
    file = open(f"{loading_path}.pickle", "rb")
    return pickle.load(file)


def check_scrapping_task(task, config: ConfigTree) -> None:
    """This method tests the scrapping task. It is checked whether
    the task can scrape the images and whether the result are actually
    image-filled folders.

    :param task: The task we would like to do
    :type task: self-written class
    :param config: Configuration file for the class
    :type config: ConfigTree
    """

    # Initiate task and run it
    task = task(config=config, re_scrape_data=False)
    task.run()

    # Checking whether the every number in the dataframe has a corresponding image
    path_config = config.get_config("paths").get_config(task.name)
    path_output = path_config.get_config("path_output")

    image_path = path_output.get_string("image_data")
    meta_df_path = path_output.get_string("processed_meta_information")

    meta_df = load_pickle(meta_df_path)
    image_number_list = meta_df.loc[:, "number"].tolist()
    meta_df_images = sorted([f"athlete_{x}.png" for x in image_number_list])
    sorted_images = sorted(os.listdir(image_path))

    assert (
        meta_df_images == sorted_images
    ), "We have a mismatch between meta information and images"

    # Checking that we do not have any missing values
    assert meta_df.isna().sum().sum() == 0, "We have missing observations"

    # Checking age for sensibility
    age_min = meta_df.loc[:, "age"].min()
    age_max = meta_df.loc[:, "age"].max()
    assert age_min >= 0 and age_max <= 100, "The age range seems questionable"


def check_preprocessing(task, config: ConfigTree) -> None:
    """This method checks the image preprocessing task

    :param task: Image classification task
    :type task: self-written class
    :param config: Corresponding Configuration file
    :type config: ConfigTree
    """

    # Initiate task and run it
    task = task(config=config)
    task.run()

    # Getting testing paths ready
    path_config = config.get_config("paths").get_config(task.name)
    path_input = path_config.get_config("path_input")
    path_output = path_config.get_config("path_output")


def check_image_classifer(task, config: ConfigTree) -> None:
    """This method checks the image classification task

    :param task: Image classification task
    :type task: self-written class
    :param config: Corresponding Configuration file
    :type config: ConfigTree
    """

    # Initiate task and run it
    task = task(config=config)
    task.run()

    # Getting testing paths ready
    path_config = config.get_config("paths").get_config(task.name)
    path_output = path_config.get_config("path_output")
