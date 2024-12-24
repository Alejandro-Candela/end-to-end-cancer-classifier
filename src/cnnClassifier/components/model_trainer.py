import os
from pathlib import Path
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    """
    The `Training` class handles the training process for a deep learning model.
    It includes functionalities for loading the base model, preparing training and validation data,
    training the model, and saving the trained model.

    Args:
        config (TrainingConfig): Configuration object containing parameters for training, such as
                                  image size, batch size, data paths, and number of epochs.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the Training class with a configuration object.

        Args:
            config (TrainingConfig): Contains parameters such as image size, batch size, training data path,
                                     number of epochs, and model save path.
        """
        self.config = config

    def get_base_model(self):
        """
        Load the base model from the specified file path in the configuration.

        This method sets the base model as `self.model` for subsequent training.
        """
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):
        """
        Prepare training and validation data generators.

        This method uses TensorFlow's `ImageDataGenerator` to preprocess images and optionally
        apply data augmentation for the training dataset. Validation data is prepared without augmentation.

        Sets the following attributes:
            - `self.train_generator`: Generator for training data.
            - `self.valid_generator`: Generator for validation data.
        """
        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.20)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the given model to the specified file path.

        Args:
            path (Path): Path where the model will be saved.
            model (tf.keras.Model): The TensorFlow model to be saved.
        """
        model.save(path)

    def train(self):
        """
        Train the model using the prepared training and validation data.

        This method calculates the steps per epoch and validation steps based on the
        number of samples and batch size, then trains the model for the specified number of epochs.
        After training, the model is saved to the path defined in the configuration.

        Sets the following attributes:
            - `self.steps_per_epoch`: Number of steps to run for each epoch.
            - `self.validation_steps`: Number of steps for validation at the end of each epoch.
        """
        self.steps_per_epoch = (
            self.train_generator.samples // self.train_generator.batch_size
        )
        self.validation_steps = (
            self.valid_generator.samples // self.valid_generator.batch_size
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)
