from pathlib import Path
import tensorflow as tf
import mlflow
import mlflow.keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import os
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json


class Evaluation:
    """
    The `Evaluation` class is designed to evaluate a pre-trained deep learning model on a validation dataset.
    It includes functionalities for loading the model, generating validation data, evaluating performance,
    saving metrics, generating a confusion matrix, and logging results into MLflow.

    Args:
        config (EvaluationConfig): Configuration object containing parameters for evaluation, including
                                   image size, batch size, data paths, and MLflow URI.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        """
        Create a validation data generator to load and preprocess the validation data.
        Uses ImageDataGenerator for rescaling pixel values and splitting the data.
        """

        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.30)

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

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Load a pre-trained TensorFlow model from a specified file path.

        Args:
            path (Path): Path to the saved model.

        Returns:
            tf.keras.Model: The loaded TensorFlow model.
        """
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """
        Perform the evaluation of the model using the validation dataset.
        This includes calculating loss and accuracy, saving scores, and generating a confusion matrix.
        """
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()
        self._save_confusion_matrix()

    def save_score(self):
        """
        Save the evaluation scores (loss and accuracy) to a JSON file for reference.
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("artifacts/evaluation/scores.json"), data=scores)

    def _save_confusion_matrix(self):
        """
        Generate and save a confusion matrix for the validation predictions.
        Also creates a visualization of the confusion matrix and stores it as a PNG file.
        """
        y_true = self.valid_generator.classes
        y_pred = np.argmax(self.model.predict(self.valid_generator), axis=1)
        class_labels = list(self.valid_generator.class_indices.keys())

        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues)

        cm_image_path = "artifacts/evaluation/confusion_matrix.png"
        plt.savefig(cm_image_path)
        plt.close()

    def log_into_mlflow(self):
        """
        Log the evaluation metrics and model parameters into MLflow.
        Registers the model if a model registry is available.
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                mlflow.keras.log_model(
                    self.model, "model", registered_model_name="VGG16Model"
                )
            else:
                mlflow.keras.log_model(self.model, "model")
