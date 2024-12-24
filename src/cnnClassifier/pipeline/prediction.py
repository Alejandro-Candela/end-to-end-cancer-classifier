import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    """
    The `PredictionPipeline` class handles the prediction process for image classification.
    It takes an image file as input, processes it, and uses a pre-trained model to predict the class of the image.

    Args:
        filename (str): The path to the image file to be classified.
    """
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        """
        Perform prediction on the input image using a pre-trained model.

        The method loads the model from a predefined path, preprocesses the input image to match the model's
        requirements, and predicts the class of the image. It returns the predicted class as a string.

        Returns:
            list[dict]: A list containing a dictionary with the prediction result.
                        The key is "image" and the value is the predicted class.

        Example:
            [{'image': 'Adenocarcinoma Cancer'}] or [{'image': 'Normal'}]
        """
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(f"result: {result}")

        if result[0] == 1:
            prediction = 'Adenocarcinoma Cancer'
            return {prediction}
        else:
            prediction = 'Normal'
            return {prediction}
