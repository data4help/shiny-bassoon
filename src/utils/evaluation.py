# %% Packages

import os
import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import seaborn as sns
import glob
from sklearn.metrics import confusion_matrix

# %% Functions


def plot_model_training_performance(
    model_dict: Dict[str, float], fname: str, metric: str
) -> None:
    """This method plots the model performance. More specifically, it plots the loss
    for the trainings-data and validation-data as well as the binary accuracy.

    :param model_dict: Dictionary containing the model performance
    :type model_dict: Dict[str, float]
    :param history: The metric we minimized during training
    :type history: Metrics
    :param fname: Name of the task from which the history comes from
    :type fname: str
    """

    fig, axs = plt.subplots(figsize=(10, 5), ncols=2)

    axs[0].plot(model_dict["loss"], label="Train")
    axs[0].plot(model_dict["val_loss"], label="Validation")
    axs[0].axhline(model_dict["test_loss"], label="Testing", linestyle="--")
    axs[0].set_title("Loss")

    axs[1].plot(model_dict[metric], label="Train")
    axs[1].plot(model_dict[f"val_{metric}"], label="Validation")
    axs[1].axhline(model_dict[f"test_{metric}"], label="Testing", linestyle="--")
    axs[1].set_title(metric.capitalize())
    axs[1].legend()

    fig.savefig(fname=fname, bbox_inches="tight")


def plot_holistic_confusion_matrix(
    fname: str, y_pred: np.array, y_true: np.array
) -> None:
    """This function plots the holistic confusion matrix. With holistic we mean that
    the confusion matrix is plotted with multi-label

    :param fname: Name under which we save the plot
    :type fname: str
    :param y_pred: String prediction
    :type y_pred: np.array
    :param y_true: String prediction truth
    :type y_true: np.array
    """

    labels = np.unique(y_true)
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    conf_df = pd.DataFrame(data=conf_matrix, columns=labels, index=labels)

    sum_over_all_predicted = conf_df.sum(axis=1)
    relative = conf_df.divide(sum_over_all_predicted, axis=0) * 100

    assert np.all(
        round(relative.sum(axis=1)) == 100
    ), "There are problems in the confusion matrix"

    fig, axs = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        data=round(relative),
        annot=True,
        ax=axs,
        vmin=0,
        vmax=100,
        cmap="viridis",
        cbar_kws={"label": "Percent of True Labels"},
    )
    axs.tick_params(axis="x", rotation=90)
    axs.set_xlabel("Predicted Label")
    axs.set_ylabel("True Label")
    axs.tick_params(axis="y", rotation=0)
    fig.savefig(fname=fname, bbox_inches="tight")


def plot_performance_barplots(fname: str, performance_dict: Dict[str, float]) -> None:
    """This functions plots the performances of the model which are found in the
    provided dictionary.

    :param fname: Name of the path the file should be saved under
    :type fname: str
    :param performance_dict: Dictionary containing various metrices
    :type performance_dict: Dict[str, float]
    """

    df = pd.DataFrame.from_dict(performance_dict, orient="index").reset_index()
    df.loc[:, "index"] = df.loc[:, "index"].str.capitalize()

    fig, axs = plt.subplots(figsize=(10, 5))
    sns.barplot(x="index", y=0, data=df, ax=axs)
    axs.set_xlabel("Metric")
    axs.set_ylabel("Score")
    fig.savefig(fname=fname, bbox_inches="tight")


def plot_positive_and_negatives(
    fname: str, y_pred: np.array, y_true: np.array, image_paths: List[str]
) -> None:

    boolean_positives = y_pred == y_true
    minimum_number = pd.Series(boolean_positives).value_counts().min()

    positive_paths = [x for x, y in zip(image_paths, boolean_positives) if y]
    positive_labels = [x for x, y in zip(y_true, boolean_positives) if y]

    negative_paths = [x for x, y in zip(image_paths, boolean_positives) if not y]
    negative_pred = [x for x, y in zip(y_pred, boolean_positives) if not y]
    negative_true = [x for x, y in zip(y_true, boolean_positives) if not y]

    os.mkdir("tmp")
    for i in range(minimum_number):
        fig, axs = plt.subplots(figsize=(10, 5), ncols=2)

        axs[0].imshow(np.array(Image.open(positive_paths[i])))
        axs[0].set_title(
            f"Prediction: {positive_labels[i]} \n True: {positive_labels[i]}"
        )
        axs[0].set_axis_off()

        axs[1].imshow(np.array(Image.open(negative_paths[i])))
        axs[1].set_title(f"Prediction: {negative_pred[i]} \n Truth: {negative_true[i]}")
        axs[1].set_axis_off()

        tmp_fname = os.path.join("tmp", f"{i}.png")
        fig.savefig(fname=tmp_fname, bbox_inches="tight")

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(f"tmp/*.png"))]
    img.save(
        fp=fname,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=2000,
        loop=0,
    )
    shutil.rmtree("tmp")
