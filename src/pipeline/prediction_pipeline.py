# %% Packages

import os
import copy
from tensorflow.keras import Model
from pyhocon import ConfigTree
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import DataFrameIterator
from tensorflow.keras.callbacks import History
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    accuracy_score,
)
from keras.utils.layer_utils import count_params

from src.base_classes.ml_pipeline import MLPipeline
from src.model.tl_models import TransferLearningModels
from src.utils.callbacks import load_callbacks
from src.utils.logging import get_logger
from src.utils.evaluation import (
    plot_model_training_performance,
    plot_holistic_confusion_matrix,
    plot_performance_barplots,
    plot_positive_and_negatives,
)

# %% Logger

logger = get_logger()

# Class


class PredictionPipeline(MLPipeline):
    def __init__(
        self,
        name: str,
        parameters: ConfigTree,
    ) -> None:
        """This init takes three arguments. The configuration file for the model,
        the prediction type which is either classification or regression and the boolean
        which decides whether we re-train the model or whether we simply load the model
        from MLFlow.

        :param name: Name of the training task
        :type name: str
        :param parameters: Parameters of the model
        :type parameters: ConfigTree
        :param path_input: Input paths
        :type path_input: ConfigTree
        :param path_output: Output paths
        :type path_output: ConfigTree
        """

        super().__init__()
        self.name = name
        self.parameters = parameters
        self.test_file_paths = None
        self.signature = None
        self.model = None
        self.test_dict = None
        self.run_id = None
        self.model_performance = None
        self.prediction_type = parameters.get_string("prediction_type")
        self.metrics = parameters.get_string("metrics")

        if self.parameters.get_bool("restore"):
            self.model = self.load_model()
        else:
            self.model = self.create_model()

        logger.info("Initial Model information:")
        self.model.summary(print_fn=logger.info)

    def create_model(self) -> Model:
        """This method builds and compiles the model for the specified prediction
        type before returning it. For now we have the option to either create
        and train a classification or a regression model.

        :return: Transfer learning model
        :rtype: Model
        """

        tl_models = TransferLearningModels(parameters=self.parameters)
        if self.prediction_type == "classification":
            model = tl_models.load_classification_model()

        elif self.prediction_type == "regression":
            model = tl_models.load_regression_model()

        logger.info(f"Created {self.prediction_type} model and compiled it")
        return model

    def compile_model(self, learning_rate: float) -> None:
        """This method compiles the model. The model is flexible in a way that the
        learning rate can be inputted into the model. That is done by first retrieving
        the optimizer by name and then set the learning rate using the input.

        :param learning_rate: Learning rate for the optimizer
        :type learning_rate: float
        """

        optimizer_name = self.parameters.get_string("optimizer")
        optimizer = tf.keras.optimizers.get(optimizer_name)
        setattr(optimizer, "lr", learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=self.parameters.get_string("loss"),
            metrics=self.metrics,
        )

    def train(
        self,
        train_gen: DataFrameIterator,
        val_gen: DataFrameIterator,
    ) -> None:
        """This method trains the prepared model. It is doing so by first loading
        all the necessary parameters from the configuration files. The model
        training is conducted using data generators. We provide one generator
        respectively for training and validation. Lastly we extract the model's
        signature which comes handy later when we store the model using MLFlow.

        :param train_gen: Data generator with the trainings-data
        :type train_gen: DataFrameIterator
        :param val_gen: Data generator with the validation-data
        :type val_gen: DataFrameIterator
        """

        logger.info("Loading parameters/ callbacks for training")
        initial_epochs = self.parameters.get_int("initial_epochs")
        fine_tune_epochs = self.parameters.get_int("fine_tune_epochs")
        monitor = self.parameters.get_string("monitor")
        patience = self.parameters.get_float("patience")

        callbacks = load_callbacks(name=self.name, monitor=monitor, patience=patience)
        steps_per_epoch = np.ceil(train_gen.n / train_gen.batch_size)
        validation_steps = np.ceil(val_gen.n / val_gen.batch_size)

        initial_epochs = 1
        fine_tune_epochs = 1
        steps_per_epoch = 1
        validation_steps = 1

        logger.info("Compile the model")
        learning_rate = self.parameters.get_float("learning_rate")
        self.compile_model(learning_rate)

        logger.info(f"Train model with {initial_epochs} epochs")
        history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=initial_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("Unfreeze the entire model")
        learning_rate_fine_tune = self.parameters.get_float("learning_rate_fine_tune")
        self.unfreeze_layers()
        self.compile_model(learning_rate_fine_tune)

        logger.info(f"Fine-tune the model with {fine_tune_epochs} epochs")
        total_epochs = fine_tune_epochs + history.epoch[-1]
        history_fine = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=total_epochs,
            initial_epoch=history.epoch[-1],
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("Gather training results - loss and metrics")
        self.gather_training_results(history, history_fine)

        logger.info("Extract the model's signature and store it")
        self.signature = self.extract_signature(val_gen)

    def unfreeze_layers(self) -> None:
        """This method unfreezes a certain percentage of the pre-trained model, in
        order to fine-tune. Which percentage to take is a bit of a trial an error
        approach.
        """

        prefreeze_trainable_weights = count_params(self.model.trainable_weights)

        self.model.get_layer("vgg16").trainable = True
        fine_tune_at_percent = self.parameters.get_float("fine_tune_at_percent")
        number_of_layers = len(self.model.get_layer("vgg16").layers)
        fine_tune_at = int(fine_tune_at_percent * number_of_layers)

        for layer in self.model.get_layer("vgg16").layers[:fine_tune_at]:
            layer.trainable = False
        postfreeze_trainable_weights = count_params(self.model.trainable_weights)

        logger.info(
            f"Before unfreezing we had {prefreeze_trainable_weights} trainable"
            f"weights, after unfreezing we have {postfreeze_trainable_weights} weights"
        )
        self.parameters.update(
            {
                "pre-freeze_trainable_weights": prefreeze_trainable_weights,
                "post_freeze_trainable_weights": postfreeze_trainable_weights,
            }
        )

    def predict(self, test_gen: DataFrameIterator) -> Tuple[np.array, np.array]:
        """This method creates predictions from the test data generator using the
        trained model. We furthermore also return the true target values.

        :param test_gen: Test data generator
        :type test_gen: DataFrameIterator
        :return: Prediction and true target
        :rtype: Tuple[np.array, np.array]
        """

        # Storing file paths
        self.test_file_paths = test_gen.filepaths

        # Evaluation
        steps = np.ceil(test_gen.n / test_gen.batch_size)
        test_dict = self.model.evaluate(test_gen, steps=steps, return_dict=True)

        if self.parameters.get_bool("restore") is False:
            self.model_performance.update(
                {f"test_{x}": y for x, y in test_dict.items()}
            )

        # Predictions
        y_pred = self.model.predict(test_gen, verbose=1, steps=steps)
        y_true = test_gen.labels

        return (y_pred, y_true)

    def evaluate(
        self,
        figure_path: str,
        y_pred: np.array,
        y_true: np.array,
        bool_plot_train_performance: bool = True,
        bool_plot_confusion_matrix: bool = True,
        bool_plot_performance_chart: bool = True,
        bool_plot_positive_and_negatives: bool = True,
    ) -> None:
        """The evaluation method plots the generally important plots, such as the model
        performance over time as well as a holistic confusion matrix.

        :param figure_path: The path in which we save the images
        :type figure_path: str
        :param y_pred: The prediction target, which is already decoded
        :type y_pred: np.array
        :param y_true: The prediction truth, also decoded
        :type y_true: np.array
        :param bool_plot_train_performance: Whether to plot that, defaults to True
        :type bool_plot_train_performance: bool, optional
        :param bool_plot_confusion_matrix: Whether to plot that, defaults to True
        :type bool_plot_confusion_matrix: bool, optional
        :param bool_plot_performance_chart: Whether to plot that, defaults to True
        :type bool_plot_performance_chart: bool, optional
        :param bool_plot_positive_and_negatives: Whether to plot that, defaults to True
        :type bool_plot_positive_and_negatives: bool, optional
        """

        if bool_plot_train_performance:
            logger.info("Plot the performance of the training")
            fname = os.path.join(figure_path, "training_summary.png")
            plot_model_training_performance(
                fname=fname,
                model_dict=self.model_performance,
                metric=self.metrics,
            )

        if bool_plot_confusion_matrix:
            logger.info("Plot holistic confusion matrix")
            fname = os.path.join(figure_path, "confusion_matrix.png")
            plot_holistic_confusion_matrix(fname=fname, y_pred=y_pred, y_true=y_true)

        if bool_plot_performance_chart:
            logger.info("Calculating the weighted accuracy score")
            precision, recall, fbeta, _ = precision_recall_fscore_support(
                y_true=y_true, y_pred=y_pred, average="weighted"
            )
            balanced_accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
            accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
            performance_dict = {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
            }
            fname = os.path.join(figure_path, "performance.png")
            plot_performance_barplots(fname=fname, performance_dict=performance_dict)
            self.model_performance.update(performance_dict)

        if bool_plot_positive_and_negatives:
            logger.info("Showcase positive and negative examples of the model")
            fname = os.path.join(figure_path, "pos_neg.gif")
            plot_positive_and_negatives(
                fname=fname,
                y_pred=y_pred,
                y_true=y_true,
                image_paths=self.test_file_paths,
            )

    def gather_training_results(self, history: History, history_fine: History) -> None:
        """This method gathers all loss and metrics results from the base training
        as well as from the fine-tuning. Afterwards, all these results are saved
        in one attribute dictionary.

        :param history: History of the model training for the base model
        :type history: History
        :param history_fine: History of the model training for the fine-tuning
        :type history_fine: History
        """

        self.model_performance = {
            "loss": history.history["loss"] + history_fine.history["loss"],
            "val_loss": history.history["val_loss"] + history_fine.history["val_loss"],
            f"{self.metrics}": history.history[f"{self.metrics}"]
            + history_fine.history[f"{self.metrics}"],
            f"val_{self.metrics}": history.history[f"val_{self.metrics}"]
            + history_fine.history[f"val_{self.metrics}"],
        }

    def extract_params_and_metrics(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """This method gathers all kinds of metrics and parameters for the model
        training. For the metrics we are gathering the loss metrics which were
        already used during model training. Further, we are only extracting the last
        value of the metrics since we are not interested in the time-series of the
        loss.

        :return: Dictionaries containing the metrics and parameters.
        :rtype: Tuple[Dict[str, float], Dict[str, float]]
        """

        # Creating metrics dictionary
        adj_metrics_dict = {}
        for key, value in self.model_performance.items():
            if isinstance(value, list):
                adj_metrics_dict[key] = value[-1]
            else:
                adj_metrics_dict[key] = value

        # Creating parameter dictionary
        params_config = copy.deepcopy(self.parameters)
        params_config.pop("augmentation_settings", None)
        augmentation_settings_dict = {
            f"augmentation_{x}": y
            for x, y in self.parameters.get_config("augmentation_settings").items()
        }
        params_config.update(augmentation_settings_dict)
        params_dict = dict(params_config)

        return adj_metrics_dict, params_dict
