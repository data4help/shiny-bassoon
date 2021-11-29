# %% Packages

import numpy as np
import pandas as pd
from typing import Tuple
from pyhocon import ConfigTree
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.keras.preprocessing.image import DataFrameIterator

from src.base_classes.task import MLTask
from src.utils.logging import get_logger

# %% Logger

logger = get_logger()


# %% Code


class TaskClassificationMultiLabel(MLTask):

    name = "task_train_classification_multilabel"
    dependencies = ["task_preprocess_images"]

    def __init__(self, config: ConfigTree) -> None:
        super().__init__(config, self.name)

    def run(self):

        logger.info("Create image loader")
        train_gen, val_gen, test_gen = self.create_image_loader()
        self.plot_augmentation_examples(train_gen=train_gen, val_gen=val_gen)

        logger.info("Load the model")
        pipeline = self.load_model()

        logger.info("Train the model")
        pipeline.train(train_gen, val_gen)

        logger.info("Create predictions for test data")
        y_pred, y_true = pipeline.predict(test_gen)
        adj_y_pred, adj_y_true = self.adjust_predictions(y_pred, y_true)

        logger.info("Evaluate the model's performance")
        pipeline.evaluate(
            figure_path=self.figure_path, y_pred=adj_y_pred, y_true=adj_y_true
        )

        logger.info("Extract metrics and parameters")
        metrics_dict, params_dict = pipeline.extract_params_and_metrics()

        logger.info("Save model and log results in MlFlow")
        pipeline.log_and_save_model(
            figure_path=self.figure_path,
            metrics_dict=metrics_dict,
            params_dict=params_dict,
            model_path=self.model_path,
        )

    def adjust_predictions(
        self, y_pred: np.array, y_true: np.array
    ) -> Tuple[np.array, np.array]:
        """This method decodes the predictions and their true counterpart and also
        brings them into a format which resembles also the initial process at the
        beginning.

        :param y_pred: Predictions of the model
        :type y_pred: np.array
        :param y_true: Truths of the testing data
        :type y_true: np.array
        :return: Two numpy arrays, containing the decoded truth and label
        :rtype: Tuple[np.array, np.array]
        """

        # Decoding the true labels
        encoder = self.load_pickle(self.encoder_path)
        decoded_y_true = encoder.inverse_transform(y_true)
        org_y_true = np.array(["_".join(x) for x in decoded_y_true])

        # Decoding the predictions
        top_two_threshold = len(y_pred[0]) - 2
        two_max_predictions = (y_pred.argsort().argsort() >= top_two_threshold).astype(
            int
        )
        assert np.all(
            two_max_predictions.sum(axis=1) == 2
        ), "There are more or less than two labels equal to one"

        decoded_y_pred = encoder.inverse_transform(two_max_predictions)
        org_y_pred = np.array(["_".join(x) for x in decoded_y_pred])

        return (org_y_pred, org_y_true)

    def encode_label(self, data: pd.DataFrame) -> pd.DataFrame:
        """This method adjusts the dataframe to fit a multi-label case. For doing
        that we first encode the target into a list of tuples which then are fed
        into a multi-label binarizer. Afterwards

        :param data: Dataframe containing targets and image paths
        :type data: pd.DataFrame
        :return: The very same as the input
        :rtype: pd.DataFrame
        """

        # Bring the target into the right format
        multi_label_target = np.array(
            [tuple(x.split("_")) for x in data.loc[:, "target"]]
        )

        # Fit, transform and save binarizer
        mlb = MultiLabelBinarizer()
        mlb.fit(multi_label_target)
        self.save_pickle(saving_path=self.encoder_path, file=mlb)

        # Add the new label to the dataframe
        columns = list(mlb.classes_)
        ml_labels_list = [list(x) for x in mlb.transform(multi_label_target)]
        ml_label_df = pd.DataFrame(data=ml_labels_list, columns=columns)
        return pd.concat((ml_label_df, data), axis=1)

    def add_output_classes_to_parameters(
        self, data_generator: DataFrameIterator
    ) -> None:
        """This method adds the number of output labels to the parameters dictionary.
        This is essential, since we need that information later on in the building of
        the transfer learning model.

        :param data_generator: The data generator which creates augmented versions of
        our images.
        :type data_generator: DataFrameIterator
        """

        _, labels, _ = next(iter(data_generator))
        dict_update = {"number_of_output_classes": len(labels[0])}
        self.parameters.update(dict_update)
