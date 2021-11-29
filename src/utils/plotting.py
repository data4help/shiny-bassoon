# %% Packages

import matplotlib.pyplot as plt
from src.utils.config import load_specific_config

# %% Load the configuration file

folder_name = "./res"
config_name = "plotting.conf"
plot_config = load_specific_config(folder_name=folder_name, config_name=config_name)

# %% Settings function


def matplotlib_settings() -> None:
    font_config = plot_config.get_config("font_sizes")
    small_size = font_config.get_int("small_size")
    medium_size = font_config.get_int("medium_size")
    big_size = font_config.get_int("big_size")

    plt.rc("font", size=small_size)
    plt.rc("axes", titlesize=small_size)
    plt.rc("axes", labelsize=medium_size)
    plt.rc("xtick", labelsize=small_size)
    plt.rc("ytick", labelsize=small_size)
    plt.rc("legend", fontsize=small_size)
    plt.rc("figure", titlesize=big_size)

    color_config = plot_config.get_config("color_code")
    style = color_config.get_string("style")
    plt.style.use(style=style)
