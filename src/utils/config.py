# %% Packages

import os
from pyhocon import ConfigFactory, ConfigTree

# %% Config loading function


def load_specific_config(folder_name: str, config_name: str) -> ConfigTree:
    """This method puts together the folder name as well as the name of the
    configuration file and afterwards loads the config file.

    :param folder_name: Folder in which the configs are in
    :type folder_name: str
    :param config_name: Name of the config file
    :type config_name: str
    :return: The config file as a config tree
    :rtype: ConfigTree
    """
    config_path = os.path.join(folder_name, config_name)
    return ConfigFactory.parse_file(config_path)


def load_config(folder_name: str) -> ConfigTree:
    """This method loads the in the input specified configuration file.

    :param folder_name: The path to the configuration file
    :type folder_name: str
    :return: Returning a configuration tree
    :rtype: ConfigTree
    """

    master_config = ConfigTree()
    configuration_files = ["parameters.conf", "paths.conf"]
    for config_name in configuration_files:
        config_file = load_specific_config(folder_name, config_name)
        master_config = master_config.with_fallback(config_file)

    return master_config
