# %% Packages

import numpy as np
import pandas as pd
from typing import Tuple
from pyhocon import ConfigTree
from sklearn.preprocessing import LabelEncoder

from tensorflow.python.keras.preprocessing.image import DataFrameIterator

from src.base_classes.task import MLTask
from src.utils.logging import get_logger

# %% Logger

logger = get_logger()


# %% Code


class TaskClassificationMultiClass(MLTask):

    name = "task_train_classification_multiclass"
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
        decoded_y_pred = encoder.inverse_transform(y_pred.argmax(axis=1))
        return (decoded_y_pred, decoded_y_true)

    def encode_label(self, data: pd.DataFrame) -> pd.DataFrame:
        """This method adjusts the dataframe to fit a multi-class case. This is done
        through one-hot encoding the target. That means that we have a target which
        is as long as we have categories. All entries are zero, except the one for
        which the target is true.

        :param data: Dataframe containing targets and image paths
        :type data: pd.DataFrame
        :return: Dataframe containing the encoded target
        :rtype: pd.DataFrame
        """

        # Bring the target into the right format
        target = data.loc[:, "target"]
        encoder = LabelEncoder()
        encoder.fit(target)
        self.save_pickle(saving_path=self.encoder_path, file=encoder)

        # Attach transformed target to dataframe
        data.loc[:, "encoded_target"] = encoder.transform(target)
        return data

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

        encoder = self.load_pickle(self.encoder_path)
        number_of_classes = len(encoder.classes_)
        dict_update = {"number_of_output_classes": number_of_classes}
        self.parameters.update(dict_update)
