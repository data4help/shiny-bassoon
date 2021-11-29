# %% Packages

import os
import shutil
import glob
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyhocon import ConfigTree
from typing import List, Tuple
from sklearn.utils.class_weight import compute_sample_weight

from src.base_classes.task import PreProcessing
from src.utils.logging import get_logger

# %% Logger

logger = get_logger()

# %% Code


class TaskPreprocessClassification(PreProcessing):

    name = "task_preprocess_classification"
    dependencies = ["task_scrapping_images"]

    def __init__(self, config: ConfigTree) -> None:
        super().__init__(config, self.name)

    def run(self):

        logger.info("Loading the meta information")
        meta_df_path = self.path_input.get_string("processed_meta_information")
        meta_df = self.load_pickle(meta_df_path)

        logger.info("Create the target feature")
        meta_df.loc[:, "target"] = self.create_target(meta_df)

        logger.info("Checking the imbalance")
        adj_meta_df = self.adjust_imbalance(meta_df)

        logger.info("Create the image paths from the image number")
        adj_meta_df_w_path = self.add_image_path(adj_meta_df)

        logger.info("Adding the sample weight to the dataframe")
        adj_meta_df_w_path_weight = self.add_sample_weight(adj_meta_df_w_path)

        logger.info("Save the processed dataframe")
        final_df = adj_meta_df_w_path_weight.loc[:, ["target", "paths", "weight_col"]]
        dataframe_path = self.path_output.get_string("clf_processed")
        self.save_pickle(saving_path=dataframe_path, file=final_df)

        logger.info("Plot example images with label")
        self.plot_label_examples(final_df)

    def add_sample_weight(self, data: pd.DataFrame) -> pd.DataFrame:
        """This method extracts the appropriate sample weight in order to counter
        the target imbalance. These sample weights are of course only used during the
        training phase of the model and neither in the predictions nor evaluation.

        :param data: Dataframe without the sample weights
        :type data: pd.DataFrame
        :return: Dataframe with sample weights
        :rtype: pd.DataFrame
        """

        y = data.loc[:, "target"]
        data.loc[:, "weight_col"] = compute_sample_weight(class_weight="balanced", y=y)
        df_weight = data.loc[:, ["target", "weight_col"]].drop_duplicates()

        fname = self.get_figure_path("sample_weights")
        fig, axs = plt.subplots(figsize=(10, 5))
        axs.bar(x=df_weight.loc[:, "target"], height=df_weight.loc[:, "weight_col"])
        axs.set_xlabel("Categories")
        axs.set_ylabel("Sample weight")
        axs.tick_params(axis="x", rotation=90)
        fig.savefig(fname=fname, bbox_inches="tight")

        return data

    def adjust_imbalance(self, data: pd.DataFrame) -> pd.DataFrame:
        """This method checks and plots the imbalance of the target value. Afterwards,
        it deletes all observations which belong to a minority class. A minority class
        is defined by having fewer observations than the in the configuration file
        specified threshold

        :param data: Dataframe containing the target columns, containing all
        observations.
        :type data: pd.DataFrame
        :return: Dataframe containing only the observations which occur more often
        than the specified threshold.
        :rtype: pd.DataFrame
        """

        logger.info("Plotting the imbalance of the target categories")
        continent_list = self.parameters.get_list("continent_list")
        target = data.loc[:, "target"]
        self.plot_imbalance(target, continent_list)

        logger.info("Extracting majority categories")
        category_boolean = data.loc[:, "continent"].isin(continent_list)

        return data.loc[category_boolean, :].reset_index(drop=True)

    def get_majority_boolean(
        self, multi_label: List[Tuple[str]], threshold_value: int
    ) -> List[bool]:
        """This method checks which multi-target has more observations than the
        threshold.

        :param multi_label: List of multi-label target
        :type multi_label: List[Tuple[str]]
        :param threshold_value: Threshold value above which the number of occurences
        have to be, in order to pass
        :type threshold_value: int
        :return: A list of booleans which category is above the threshold
        :rtype: List[bool]
        """

        target_above_thresh = pd.Series(multi_label).value_counts() > threshold_value
        return [target_above_thresh[x] for x in multi_label]

    def plot_imbalance(self, label: List[str], continent_list: List[str]) -> None:
        """This method plots the categorical features we use as the target. All
        categories which are used, are shown in a different color, compared to the ones
        we exclude from this experiment.

        :param label: List of strings representing the multi-label
        :type label: List[str]
        :param continent_list: List of continents which are wanted in the data
        :type continent_list: List[str]
        """

        fname = self.get_figure_path("label_imbalance")
        df = label.value_counts().reset_index().rename(columns={"index": "category"})
        df.loc[:, "included"] = df.loc[:, "category"].apply(
            lambda x: x.split("_")[1].title() in continent_list
        )

        fig, axs = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x="category", y="target", hue="included", data=df, ax=axs, dodge=False
        )
        axs.set_xlabel("Categories")
        axs.set_ylabel("# of observations")
        axs.tick_params(axis="x", rotation=90)
        axs.legend(title="Included in the Classification Model")
        fig.savefig(fname=fname, bbox_inches="tight")

    def add_image_path(self, data: pd.DataFrame) -> pd.DataFrame:
        """This method adds a prefix and a suffix to the image number column. This
        represents the path under which the image can be found the observation
        describes. This is necessary in order for the model later to load
        the image observation

        :param data: Dataframe containing the image number.
        :type data: pd.DataFrame
        :return: Dataframe containing no number anymore, but the entire path to the
        image/
        :rtype: pd.DataFrame
        """

        image_folder_path = self.path_input.get_string("image_data")
        general_image_prefix = os.path.join(image_folder_path, "athlete_")
        general_image_suffix = ".png"

        data.loc[:, "paths"] = (
            general_image_prefix
            + data.loc[:, "number"].astype(str)
            + general_image_suffix
        ).values

        return data.drop(columns=["number"])

    def create_target(self, data: pd.DataFrame) -> List[str]:
        """This method combines all columns which contain a part of the multi-label
        into a list of strings containing the multi-target. The categories are
        separated by an underscore.

        :param data: Dataframe containing the multi-label features.
        :type data: pd.DataFrame
        :return: List of string containing the multi-label
        :rtype: List[str]
        """

        multi_label_columns = self.parameters.get_list("target_columns")
        return (
            data.loc[:, multi_label_columns]
            .apply(lambda x: "_".join(x.str.lower()), axis=1)
            .tolist()
        )

    def plot_label_examples(self, data: pd.DataFrame) -> None:
        """This method plots example images of the target into a gif

        :param data: Dataframe containing the labels and image paths
        :type data: pd.DataFrame
        """

        unique_encoded_labels = list(set(data.loc[:, "target"]))
        dict_df = {
            x: data.query(f"target==@x").reset_index(drop=True)
            for x in unique_encoded_labels
        }
        min_number_of_images = np.min([len(dict_df[x]) for x in unique_encoded_labels])

        tmp_folder = os.path.join(self.figure_path, "tmp")
        os.mkdir(tmp_folder)

        logger.info("Create image used in an example gif")
        for i in range(min_number_of_images):
            fig, axs = plt.subplots(figsize=(20, 5), ncols=len(unique_encoded_labels))
            axs = axs.ravel()
            for j, (key, value) in enumerate(dict_df.items()):
                axs[j].imshow(np.array(Image.open(value.loc[i, "paths"])))
                axs[j].set_title(key)
                axs[j].set_axis_off()
            fname = os.path.join(tmp_folder, f"{i}.png")
            fig.savefig(fname=fname, bbox_inches="tight")

        fname = self.get_figure_path("gif_example")
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(f"{tmp_folder}/*.png"))]
        img.save(
            fp=fname,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=200,
            loop=0,
        )
        shutil.rmtree(tmp_folder)
