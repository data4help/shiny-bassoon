# %% Packages

import os
import logging
from logging import Logger
import logging.config

# %% Functions


def get_logger() -> Logger:
    """This function creates a new log file

    :return: Logger with the specifications from the config
    :rtype: Logger
    """

    # Create new log files
    log_file_name = os.path.join("logs", "logging", "file.log")
    logging.config.fileConfig(
        fname="res/logging.conf",
        defaults={"logfilename": log_file_name},
        disable_existing_loggers=False,
    )
    return logging.getLogger("potato")
