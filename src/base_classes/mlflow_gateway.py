# %% Packages

import os
from typing import Dict
import mlflow
from keras.engine.functional import Functional
from mlflow.models.signature import ModelSignature

from src.utils.logging import get_logger

# %% Logger

logger = get_logger()


# %% Class


class MLFlowGateway:
    def __init__(self, name: str) -> None:
        self.name = name
        self.mlflow_client = mlflow.tracking.MlflowClient()
        self.experiment_id = self.get_experiment()

    def get_experiment(self) -> str:
        """This method loads the experiment. Either there is an experiment already
        in existence with a certain name, or we have to create the experiment with
        the provided name. We then extract the experiment id with which we can
        create runs for that experiment.

        :return: Returning the experiment Id
        :rtype: str
        """

        experiment_id = self.mlflow_client.get_experiment_by_name(self.name)
        if experiment_id is None:
            experiment_id = self.mlflow_client.create_experiment(self.name)
        else:
            experiment_id = experiment_id.experiment_id
        return experiment_id

    def log_run(
        self,
        model: Functional,
        signature: ModelSignature,
        figure_path: str,
        metrics_dict: Dict[str, float],
        params_dict: Dict[str, float],
    ) -> None:
        """This method logs the model, metrics and parameters for every run.
        Furthermore, the model signature is saved with the model. The run is triggered
        under the specific experiment id.

        :param model: The CNN which was trained priorly
        :type model: Functional
        :param signature: The model signature which describes the necessary input
        and output
        :type signature: ModelSignature
        :param figure_path: The path from which we are saving the figures from
        :type figure_path: str
        :param metrics_dict: Dictionary containing model metrics
        :type metrics_dict: Dict
        :param params_dict: Dictionary containing model parameters
        :type params_dict: Dict
        """

        logger.info(f"Creating a new run under the experiment id {self.experiment_id}")
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            mlflow.log_params(params_dict)
            mlflow.log_metrics(metrics_dict)
            mlflow.log_artifacts(figure_path, artifact_path="figures")
            mlflow.keras.log_model(
                keras_model=model,
                artifact_path="model",
                signature=signature,
            )

    def get_run(self, criterion: str, prediction_type: str) -> Functional:
        """This method retrieves a past run, using the criterion to find out which
        run to use. Afterwards we retrieve the run_id with which we then can load
        the stored keras model.

        :param criterion: Criterion by which we would like to retrieve the model
        :type criterion: str
        :param prediction_type: Only used for the logs
        :type prediction_type: str
        :raises ValueError: For the cases where we cannot retrieve any run
        :return: The previously trained keras model
        :rtype: Functional
        """

        logger.info(f"Loading the {criterion} {prediction_type} model for {self.name}")
        if criterion == "latest":
            order_criterion = "attribute.start_time DESC"
        else:
            order_criterion = f"metrics.{criterion} DESC"

        try:
            run = self.mlflow_client.search_runs(
                experiment_ids=self.experiment_id,
                order_by=[order_criterion],
                max_results=1,
            ).pop()
            run_id = run.info.run_id
        except Exception as e:
            raise ValueError(f"The MLFlow does not contain any runs - {e}")

        model_path = os.path.join("runs:", run_id, "model")
        return mlflow.keras.load_model(model_path)
