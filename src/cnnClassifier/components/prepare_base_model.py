import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """
    The `PrepareBaseModel` class is responsible for setting up a base deep learning model using the VGG16 architecture.
    It includes functionalities for loading a base model, preparing a full model for fine-tuning, and saving models.

    Args:
        config (PrepareBaseModelConfig): Configuration object containing parameters such as image size,
                                         weights, class count, and model save paths.
    """

    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        """
        Load the base model using the VGG16 architecture with parameters specified in the configuration.
        The base model is then saved to the configured path for reuse.
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepare the full model for training by adding a classifier on top of the base model.
        The base model layers can be frozen entirely or partially for transfer learning.

        Args:
            model (tf.keras.Model): The base model to extend.
            classes (int): Number of output classes for the classifier.
            freeze_all (bool): If True, freezes all layers of the base model.
            freeze_till (int or None): Number of layers to keep unfrozen from the end of the base model.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            tf.keras.Model: The complete model ready for training.
        """
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(
            flatten_in
        )

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Update the base model by adding a classifier and preparing it for training.
        The extended model is then saved to the configured path.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the given model to the specified file path.

        Args:
            path (Path): Path where the model will be saved.
            model (tf.keras.Model): The TensorFlow model to be saved.
        """
        model.save(path)