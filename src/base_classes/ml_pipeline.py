# %% Packages

from keras.engine.functional import Functional
from typing import Dict, Tuple
from abc import ABC, abstractmethod
from tensorflow.python.keras.preprocessing.image import DataFrameIterator
from mlflow.models.signature import infer_signature, ModelSignature

from src.base_classes.mlflow_gateway import MLFlowGateway

# %% Class


class MLPipeline(ABC):
    """This base-class for ML pipeline states which functionalities should
    be available to all pipelines coming from this class."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def load_model(self) -> Tuple[Functional, str]:
        """This method loads the stored model from the past run and returns it.
        Furthermore, the id from the past run is returned in order to keep track of
        which model did what

        :return: Returning the model as well as the recommendation id.
        :rtype: Tuple[Functional, str]
        """

        criterion = self.parameters.get_string("criterion")
        prediction_type = self.parameters.get_string("prediction_type")
        experiment = MLFlowGateway(name=self.name)
        return experiment.get_run(
            criterion=criterion,
            prediction_type=prediction_type,
        )

    def log_and_save_model(
        self,
        figure_path: str,
        metrics_dict: Dict,
        params_dict: Dict,
        model_path: str,
    ) -> None:
        """This method logs and saves all the results we obtained. This is done
        by first creating the experiment using the self-written MLFlowGateway. This
        class retrieves us the experiment id which we then can use to create a new
        run to which we then save our results to. We log the trained model, the metrics
        dictionary as well as the parameter dictionary.

        :param figure_path: Path which contains the images we would like to save
        :type figure_path: str
        :param metrics_dict: Dictionary containing all our metric, mostly describing
        how the performance of our model was
        :type metrics_dict: Dict
        :param params_dict: Dictionary containing all our parameters, mostly describin
        how the model was build
        :type params_dict: Dict
        :param model_path: Local path under which the model is saved
        :type model_path: str
        """

        # Save model locally
        self.model.save(model_path)

        # Log model on MLFlow
        experiment = MLFlowGateway(name=self.name)
        experiment.log_run(
            model=self.model,
            signature=self.signature,
            figure_path=figure_path,
            metrics_dict=metrics_dict,
            params_dict=params_dict,
        )

    def extract_signature(self, data_generator: DataFrameIterator) -> ModelSignature:
        """This method extracts the signature of the fitted model. A signature
        gives an indication about what shape and type the input and output are.

        :param data_generator: The data-generator used for training the model
        :type data_generator: DataFrameIterator
        :return: Signature which gives an indication about the input and output shape
        :rtype: ModelSignature
        """

        images, labels = next(iter(data_generator))
        example_image, example_label = images[0], labels[0]
        return infer_signature(example_image, example_label)
