# %% Packages

import os
import shutil
import pickle
import numpy as np
import pandas as pd
from pyhocon import ConfigTree
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import DataFrameIterator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import random

from src.utils.plotting import matplotlib_settings
from src.utils.logging import get_logger
from src.pipeline.prediction_pipeline import PredictionPipeline

# %% Logger

logger = get_logger()

# %% Classes


class Base(ABC):
    @abstractmethod
    def run(self):
        """Every task needs to have a run() function that executes
        the desired program
        """


class ConfigTask(Base):
    def __init__(self, config: ConfigTree, name: str) -> None:

        # Parameters
        self.parameters = self.extract_specific_config(
            config, type_name="parameters", task_name=name
        )

        # Paths
        paths = self.extract_specific_config(config, type_name="paths", task_name=name)
        self.path_input = paths.get_config("path_input")
        self.path_output = paths.get_config("path_output")

        # Figure pathing
        self.figure_path = os.path.join(paths.get_string("figure_path"), self.name)
        self.create_or_clean_figure_directory(self.figure_path)

        # Matplotlib settings
        matplotlib_settings()

    def get_figure_path(self, figure_name_key: str) -> str:

        figure_name = self.path_output.get_string(figure_name_key)
        return os.path.join(self.figure_path, figure_name)

    def create_or_clean_figure_directory(self, path: str) -> None:
        """This method creates the directory for the task if it does not exist. If it
        does exist, all files are cleaned/ removed.

        :param path: Name of the task which is also the file directory
        :type path: str
        """

        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)

    def extract_specific_config(
        self, *config, type_name: str, task_name: str
    ) -> ConfigTree:
        """This method extracts the specific type out of the configuration file
        and afterwards extracts the correct name. Furthermore, we find the
        default version of the configuration file and state it as the fallback

        :param type_name: States what kind of configuration we are looking for
        :type type_name: str
        :param task_name: Stating for which task we are trying to retrieve the config
        :type task_name: str
        :return: The desired config file
        :rtype: ConfigTree
        """

        config = config[0]
        type_config = config.get_config(type_name)
        fallback = config.get_config(f"{type_name}_default")
        task_config = type_config.get_config(task_name).with_fallback(fallback)
        return task_config


class Task(ConfigTask):
    def __init__(self, config: ConfigTree, name: str) -> None:
        super().__init__(config, name)

    def save_pickle(self, saving_path: str, file) -> None:
        """This method saves the given file at the specified path using pickle

        :param saving_path: Saving path
        :type saving_path: str
        :param file: Object to be saved
        :type file: Could be anything basically
        """

        logger.info(f"Saving {file} under {saving_path}")
        with open(f"{saving_path}.pickle", "wb") as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, loading_path: str):
        """This method loads the file at the specified path

        :param loading_path: Path at which object is saved
        :type loading_path: str
        :return: Desired file
        :rtype: Could be basically anything
        """

        logger.info(f"Loading file from {loading_path}")
        file = open(f"{loading_path}.pickle", "rb")
        return pickle.load(file)


class PreProcessing(Task):
    def __init__(self, config: ConfigTree, name: str) -> None:
        super().__init__(config, name)


class MLTask(Task):
    def __init__(self, config: ConfigTree, name: str) -> None:
        super().__init__(config, name)

        # Creation of encoder path
        prediction_type = self.parameters.get_string("prediction_type")
        encoder_folder = self.create_model_folder(prediction_type, "encoder")
        self.encoder_path = os.path.join(encoder_folder, "encoder")
        self.model_path = self.create_model_folder(prediction_type, "prediction_model")

    def create_model_folder(self, prediction_type: str, model_type: str) -> str:
        """This method builds the model path for the specified input. After
        the path is build we create or clean the path.

        :param prediction_type: Indicating whether it is a classification or regression
        :type prediction_type: str
        :param model_type: Indication whether it is an encoder or prediction model
        :type model_type: str
        :return: Created/ cleaned path of model
        :rtype: str
        """

        path = os.path.join("./model", prediction_type, model_type, self.name)
        self.create_or_clean_figure_directory(path)
        return path

    def load_model(self, appendix: str = None) -> PredictionPipeline:
        """This method initializes the model pipeline. For that reason we specify
        the model configurations, as well as the prediction type and whether we should
        restore the pipeline, meaning whether re-training is necessary

        :return: Returning the model class, which contains the training and evaluate
        methods
        :rtype: PredictionPipeline
        """

        name = f"{self.name}_{appendix}" if appendix else self.name
        return PredictionPipeline(
            name=name,
            parameters=self.parameters,
        )

    def shuffle_dataset(
        self, dataframe: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """This method separates the dataframe into train, test and validation set.
        This is necessary as we would like to assess the performance of our model on
        a hold-out test dataset, while assessing the performance on the validation data.

        :param dataframe: Dataframe containing all image paths and labels
        :type dataframe: pd.DataFrame
        :return: Returning the split dataframe into train and test
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        """

        train_size = self.parameters.get_float("train_size")
        random_state = self.parameters.get_int("random_state")

        train_df, test_val_df = train_test_split(
            dataframe,
            train_size=train_size,
            random_state=random_state,
            shuffle=True,
            stratify=dataframe.loc[:, "target"],
        )

        test_df, val_df = train_test_split(
            test_val_df,
            train_size=0.5,
            random_state=random_state,
            shuffle=True,
            stratify=test_val_df.loc[:, "target"],
        )

        return train_df, val_df, test_df

    def create_image_loader(
        self,
    ) -> Tuple[DataFrameIterator, DataFrameIterator, DataFrameIterator]:
        """This method uses the imagedatagenerator and creates a data loader.
        Whereas the Image data generator specified all the steps that are going
        to be applied to the images, the data frame iterator, loads the images from
        their specified paths and attaches the corresponding label to them.
        Lastly we plot some example images in order to showcase how the images look
        like after we apply the augmentation and preprocessing. Since we are needing
        one data generator for train and test, we create two of them in a loop, after
        separating the dataset through a shuffling

        :param datagen: Data generator which specifies how the images are preprocessed
        :type datagen: ImageDataGenerator
        :return: Data loader which loads the desired images and labels, train and test
        :rtype: Tuple[DataFrameIterator, DataFrameIterator]
        """

        logger.info("Loading the image dataframe which specifies label and image path")
        dataframe_path = self.path_input.get_string("clf_data")
        dataframe = self.load_pickle(dataframe_path)

        logger.info(f"Adjusting data to suit {self.name} case")
        dataframe_encoded = self.encode_label(dataframe)
        train_df, val_df, test_df = self.shuffle_dataset(dataframe_encoded)

        logger.info("Create generators for train, test and validation")
        train_gen = self.create_image_data_generator(train=True, data=train_df)
        val_gen = self.create_image_data_generator(train=False, data=val_df)
        test_gen = self.create_image_data_generator(train=False, data=test_df)

        logger.info("Add the information about how many classes we are predicting")
        self.add_output_classes_to_parameters(train_gen)

        return (train_gen, val_gen, test_gen)

    def plot_augmentation_examples(
        self, train_gen: DataFrameIterator, val_gen: DataFrameIterator
    ) -> None:
        """This method plots two example images from the validation and from the
        training data generator. One of them should show data augmentation, whereas
        the other should show none augmentation at all.

        :param train_gen: The training generator
        :type train_gen: DataFrameIterator
        :param val_gen: The validation generator
        :type val_gen: DataFrameIterator
        """

        # Extract training and validation image
        random_number = random.randint(0, train_gen.n)
        random_label = np.array([train_gen.labels[random_number].tolist()])
        random_path = train_gen.filepaths[random_number]

        target_size = self.parameters.get_list("target_size")
        img_array = np.array(Image.open(random_path).resize(target_size))[np.newaxis, :]

        # Creating the images
        train_image = next(iter(train_gen.image_data_generator.flow(x=img_array)))
        val_image = next(iter(val_gen.image_data_generator.flow(x=img_array)))

        image_array = [train_image[0], val_image[0], img_array[0]]
        subplot_names = ["Training", "Validation/ Testing", "Original"]

        encoder = self.load_pickle(self.encoder_path)
        decoded_random_label = encoder.inverse_transform(random_label)

        title = f"Class: {decoded_random_label} - Label: {random_label}"
        fname = self.get_figure_path("augmentation_examples")
        self.plot_example_image_grid(
            title=title,
            image_array=image_array,
            subplot_names=subplot_names,
            fname=fname,
        )

    def plot_example_image_grid(
        self,
        title: str,
        image_array: np.array,
        subplot_names: List[str],
        fname: str,
    ):
        """This method plots a squared image grid of example images. This plot
        is supposed to show how the labels look like with a corresponding image.

        :param title: The title of the subplots
        :type title: str
        :param image_array: Array full of images
        :type image_array: np.array
        :param subplot_names: Labels of the image
        :type subplot_names: List[str]
        :param fname: Name under which the image should be saved under.
        :type fname: str
        """

        number_of_examples = len(image_array)
        fig, axs = plt.subplots(
            figsize=(10, 5),
            ncols=number_of_examples,
        )
        fig.suptitle(title)
        axs = axs.ravel()
        for ax, image, subplot_name in zip(axs, image_array, subplot_names):
            ax.imshow(image)
            ax.set_title(subplot_name)
            ax.set_axis_off()
        fig.savefig(fname=fname, bbox_inches="tight")

    def create_image_data_generator(
        self, train: bool, data: pd.DataFrame
    ) -> DataFrameIterator:
        """This method loads the image data generator. We differentiate between whether
        we should provide augmentated images. This is necessary for training purposes
        but not for validation and testing purposes.

        :param train: Indication whether the resulting image data generator is used
        for training's purposes
        :type train: bool
        :param data: Dataframe with the paths and label columns
        :type data: pd.DataFrame
        :return: Frame iterator which generates labels and data
        :rtype: DataFrameIterator
        """

        x_col = "paths"
        y_col = [x for x in data.columns if x not in ["paths", "weight_col", "target"]]
        batch_size = self.parameters.get_int("batch_size")
        target_size = self.parameters.get_list("target_size")
        class_mode = self.parameters.get_string("class_mode")
        seed = self.parameters.get_int("random_state")

        flow_dict = {
            "dataframe": data,
            "directory": ".",
            "x_col": x_col,
            "y_col": y_col,
            "batch_size": batch_size,
            "target_size": target_size,
            "class_mode": class_mode,
            "seed": seed,
            "shuffle": False,
        }

        settings = {"preprocessing_function": preprocess_input}
        if train:
            augment_settings = self.parameters.get_config("augmentation_settings")
            settings.update(augment_settings)
            flow_dict.update({"weight_col": "weight_col"})

        return ImageDataGenerator(**settings).flow_from_dataframe(**flow_dict)
