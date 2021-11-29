# %% Packages

from src.tasks.ml.train_classification import TaskTrainClassifier
from src.tasks.etl.scrapping_images import TaskScrappingImages
from src.tasks.etl.preprocess_classification import TaskPreprocessClassification

# %% Tasks

# Tasks for the main pipeline
tasks = [
    TaskScrappingImages,
    TaskTrainClassifier,
    TaskPreprocessClassification,
]

task_index = {}
for task in tasks:
    task_index[task.name] = task
