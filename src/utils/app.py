# %% Packages

import os
import logging
from src.utils.logging import get_logger
from src.utils.config import load_config
from src.tasks.index import index

# %% Initiate logger

logger = get_logger()

# %% Code


def run_task(task: str, config_folder: str) -> None:
    """This function loads the general config file and the relevant task
    we would like to execute. For that purposes we take a look in our
    predefined index and extract the task that was specified in the cli.

    :param task: Name of the task we would like to run
    :type task: str
    :param config_folder: Path to where the config files are
    :type config_folder: str
    :raises ValueError: Error if the task we would like to train is not present
    """
    config_files = [x for x in os.listdir(config_folder) if x.endswith(".conf")]
    logger.info(f"Instanciate task {task} with configs in {config_files}.")
    config = load_config(config_folder)

    if task in index.keys():
        task_obj = index[task]
        task = task_obj(config)
        task.run()  # run the task run() function
    else:
        logger.error(f"The task '{task}' is not in the task index")
        raise ValueError(f"The task '{task}' is not in the task index.")

    logging.shutdown()
