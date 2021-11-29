# %% Packages

from typing import List
import numpy as np

# %% Functions


def calculate_binary_predictions(y_pred_proba: np.array, labels: List) -> np.array:
    """This function separates the continent information from the gender information,
    finds the maximum for the two topics individually and puts the array back
    together. That ensures that we always pick a gender and always pick a continent.

    :param y_pred_proba: Prediction probability
    :type y_pred_proba: np.array
    :param labels: Labels from the encoder
    :type labels: List
    :return: Concatenated maximum of the two topics
    :rtype: np.array
    """

    # Find position of male and female label
    genders = ["female", "male"]
    gender_position = np.array([True if x in genders else False for x in labels])
    gender_max = y_pred_proba[:, gender_position].argsort(axis=1)

    # Find all other positions and find the maximum value
    continent_position = ~gender_position
    continent_matrix = y_pred_proba[:, continent_position]
    binary_continent_matrix = np.zeros_like(continent_matrix)
    binary_continent_matrix[
        np.arange(len(continent_matrix)), continent_matrix.argmax(1)
    ] = 1

    # Putting results back together
    total_pred = np.zeros_like(y_pred_proba)
    total_pred[:, gender_position] = gender_max
    total_pred[:, continent_position] = binary_continent_matrix

    assert np.all(
        total_pred.sum(axis=1) == 2
    ), "There are more true labels then there should be"

    return total_pred.astype(int)
