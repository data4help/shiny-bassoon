# %% Packages

import argparse
from src.utils.logging import get_logger
from src.utils.app import run_task

# %% Initiate logger

logger = get_logger(remove_old_files=True)
logger.info("Mode: Command Line Interface")

# %% Code


def main():
    args = parse_args()
    execute(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Psychic-Potato")
    parser.add_argument(
        "-t",
        "--task",
        metavar="TASK",
        type=str,
        required=True,
        help="Task to execute.",
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="CONFIG",
        type=str,
        required=True,
        help="Path to environment config file for the task.",
    )

    args = parser.parse_args()
    return args


def execute(args: argparse.Namespace):
    task = args.task
    config = args.config

    logger.info(f"Running the task {task} of the application with config {config}.")
    # if instructions:
    run_task(task=task, config_path=config)
