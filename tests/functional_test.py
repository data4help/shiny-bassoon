# %% Packages

import pytest

from src.utils.config import load_config
from src.utils.logging import get_logger
from tests.utils import (
    check_scrapping_task,
    check_image_classifer,
    check_preprocessing,
)

from src.tasks.etl.preprocess_classification import TaskPreprocessClassification
from src.tasks.etl.scrapping_images import TaskScrappingImages
from src.tasks.ml.train_clf_multilabel import TaskClassificationMultiLabel
from src.tasks.ml.train_clf_multiclass import TaskClassificationMultiClass
from src.tasks.ml.train_clf_chain import TaskClassificationChain

# %% Instantiate the logger and clean log files

logger = get_logger()
logger.info("Mode: Testing environment")

# %% Load the config file

config = load_config(folder_name="res/")

# %% Tests


@pytest.mark.order(1)
def test_scrapping_images():
    check_scrapping_task(task=TaskScrappingImages, config=config)


@pytest.mark.order(1)
def test_preprocessing_classifier():
    check_preprocessing(task=TaskPreprocessClassification, config=config)


@pytest.mark.order(1)
def test_clf_multilabel():
    check_image_classifer(task=TaskClassificationMultiLabel, config=config)


@pytest.mark.order(1)
def test_clf_multiclass():
    check_image_classifer(task=TaskClassificationMultiClass, config=config)


@pytest.mark.order(1)
def test_clf_chain():
    check_image_classifer(task=TaskClassificationChain, config=config)
