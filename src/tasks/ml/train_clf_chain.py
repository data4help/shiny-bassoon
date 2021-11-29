# %% Packages

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from pyhocon import ConfigTree
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.preprocessing.image import DataFrameIterator

from src.base_classes.task import MLTask
from src.utils.logging import get_logger

# %% Logger

logger = get_logger()


# %% Code


class TaskClassificationChain(MLTask):

    name = "task_train_classification_chain"
    dependencies = ["task_preprocess_images"]

    def __init__(self, config: ConfigTree) -> None:
        super().__init__(config, self.name)
        self.target_columns = self.parameters.get_list("target_columns")
        self.target_column = None

    def run(self):

        logger.info("Prepare encoders for chained classification")
        self.create_chain_encoders()

        y_pred_dict, y_true_dict = {}, {}
        for target in self.target_columns:

            logger.info(f"The current target column is called: {target}")
            self.target_column = target

            logger.info("Adding the image folder for the task")
            target_figure_path = os.path.join(self.figure_path, target)
            self.create_or_clean_figure_directory(target_figure_path)

            logger.info(f"Create image loader for the {target} target")
            train_gen, val_gen, test_gen = self.create_image_loader()

            logger.info("Load the model")
            pipeline = self.load_model(appendix=target)

            logger.info("Train the model")
            pipeline.train(train_gen, val_gen)

            logger.info("Create predictions for test data")
            y_pred, y_true = pipeline.predict(test_gen)
            adj_y_pred, adj_y_true = self.decode_predictions(y_pred, y_true, target)
            y_pred_dict[target] = adj_y_pred
            y_true_dict[target] = adj_y_true

            logger.info(f"Evaluate the model's performance for the column {target}")
            pipeline.evaluate(
                figure_path=target_figure_path,
                y_pred=adj_y_pred,
                y_true=adj_y_true,
            )

            logger.info("Extract metrics and parameters")
            metrics_dict, params_dict = pipeline.extract_params_and_metrics()
            params_dict.update({"target_column": target})

            logger.info("Save model and log results in MlFlow")
            model_path = os.path.join(self.model_path, self.target_column)
            self.create_or_clean_figure_directory(model_path)
            pipeline.log_and_save_model(
                figure_path=target_figure_path,
                metrics_dict=metrics_dict,
                params_dict=params_dict,
                model_path=model_path,
            )

        logger.info("Adjust the predictions from the chained-classifier")
        adj_y_pred = self.concat_predictions(y_pred_dict)
        adj_y_true = self.concat_predictions(y_true_dict)

        logger.info(f"Evaluate the model's performance for the column {target}")
        pipeline.evaluate(
            figure_path=self.figure_path,
            y_pred=adj_y_pred,
            y_true=adj_y_true,
            bool_plot_train_performance=False,
        )

    def create_chain_encoders(self) -> None:
        """This method creates the encoder for the chained classification method.
        Since we will classify one item at a time, we need to create the encoders
        in before-hand, since they would be otherwise over-written in the loop.
        """

        dataframe_path = self.path_input.get_string("clf_data")
        dataframe = self.load_pickle(dataframe_path)
        target_df = (
            dataframe.loc[:, "target"]
            .str.split("_", expand=True)
            .rename(columns={i: name for i, name in enumerate(self.target_columns)})
        )

        encoder_dict = {}
        for target in self.target_columns:

            tmp_target = target_df.loc[:, target]
            tmp_encoder = OneHotEncoder()
            tmp_encoder.fit(tmp_target.values.reshape(-1, 1))
            encoder_dict[target] = tmp_encoder

        self.save_pickle(saving_path=self.encoder_path, file=encoder_dict)

    def concat_predictions(self, y_dict: Dict[str, np.array]) -> np.array:
        """This method decodes the predictions and their true counterpart and also
        brings them into a format which resembles also the initial process at the
        beginning.

        :param y_dict: Dictionary with string and numpy arrays with the predictions
        :type y_dict: Dict[str, np.array]
        """

        return np.array(list(map("_".join, zip(*y_dict.values()))))

    def decode_predictions(
        self, y_pred: np.array, y_true: np.array, target: str
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

        tmp_encoder = encoder[target]
        tmp_pred = np.zeros_like(y_pred)
        tmp_pred[np.arange(len(tmp_pred)), y_pred.argmax(1)] = 1

        assert np.all(
            tmp_pred.sum(axis=1) == 1
        ), f"The inverse transform for {target} did not work"

        adj_y_pred = np.array([x[0] for x in tmp_encoder.inverse_transform(tmp_pred)])
        adj_y_true = np.array([x[0] for x in tmp_encoder.inverse_transform(y_true)])

        return (adj_y_pred, adj_y_true)

    def encode_label(self, data: pd.DataFrame) -> pd.DataFrame:
        """This method adjusts the dataframe to fit a chained-classification case.
        For that we one-hot encode the target and make use of the fact that we have
        two binary cases to predict. Therefore, we use a one-hot-encoder

        :param data: Dataframe containing targets and image paths
        :type data: pd.DataFrame
        :return: Dataframe containing the encoded target
        :rtype: pd.DataFrame
        """

        # Bring the target into the right format
        target = data.loc[:, "target"]
        target_df = (
            target.str.split("_", expand=True)
            .rename(columns={i: name for i, name in enumerate(self.target_columns)})
            .loc[:, self.target_column]
        )

        encoder_dict = self.load_pickle(self.encoder_path)
        encoder = encoder_dict[self.target_column]

        # Attach transformed target to dataframe
        encoded_data = (
            encoder.transform(target_df.values.reshape(-1, 1))
            .toarray()
            .astype(int)
            .tolist()
        )
        encoded_df = pd.DataFrame(
            data=encoded_data,
            columns=[f"column_{i}" for i in range(len(encoded_data[0]))],
        )

        assert np.all(
            encoded_df.sum(axis=1) == 1
        ), f"The encoding in the multi-chain did not work for {self.target_column}"
        return pd.concat((encoded_df, data), axis=1)

    def add_output_classes_to_parameters(
        self, data_generator: DataFrameIterator
    ) -> None:
        """This method adds the number of output labels to the parameters dictionary.

        :param data_generator: The data generator which creates augmented versions of
        our images - Not used in this case, though must stay here since it is used
        in other child classes like this.
        :type data_generator: DataFrameIterator
        """

        _, label, _ = next(iter(data_generator))
        dict_update = {"number_of_output_classes": len(label[0])}
        self.parameters.update(dict_update)
