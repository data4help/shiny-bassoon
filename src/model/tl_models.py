# %% Packages

from keras.engine.training import Model
from tensorflow import keras
from typing import List
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras import activations

# %% Class


class TransferLearningModels:
    def __init__(self, parameters):
        self.parameters = parameters
        self.input_shape = self.load_input_shape()

    def load_input_shape(self) -> List[int]:
        """This method puts the target size together with the number of dimensions
        all specified in the configuration file. Through that method, we can easily
        change any parameter at will in the configuration file. This comes handy, if
        we for example would like to deal with gray-scale images for example.

        :return: Input shape which is build out of target size and dimensionality number
        :rtype: List[int]
        """

        target_size = self.parameters.get_list("target_size")
        number_of_dimension = self.parameters.get_int("target_dimensions")
        target_size.append(number_of_dimension)
        return target_size

    def load_classification_model(self) -> Model:
        """This method loads the pre-trained model and stacks a new head on top of it.

        :return: Keras model
        :rtype: Model
        """

        number_of_output_classes = self.parameters.get_int("number_of_output_classes")
        seed = self.parameters.get_int("random_state")
        base_model = self.load_base_model()

        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        activation = tf.keras.activations.get(self.parameters.get_string("activation"))

        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(
            number_of_output_classes,
            kernel_initializer=initializer,
            activation=activation,
        )(x)

        return keras.Model(inputs, outputs)

    def load_regression_model(self):
        base_model = self.load_base_model()

    def load_base_model(self) -> VGG16:
        """This method loads the base line model of the classification model. Herein
        we use a pre-trained VGG16 network. In order to be used in transfer-learning,
        we removed the top-layers and initialized it with the 'imagenet' weights.
        Furthermore, we set the trainability to False since the base-layers should
        not be allowed to adjust within transfer-learning.

        :return: Pre-trained model with fixed weights
        :rtype: VGG16
        """

        transfer_learning_model = VGG16(
            include_top=False, weights="imagenet", input_shape=self.input_shape
        )
        transfer_learning_model.trainable = False
        return transfer_learning_model
