import os

import joblib
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml

import ui.ui as ui


def fetch_mnist_data():
    if os.path.exists('./data/mnist_dataset.joblib'):
        ui.announce("Loading MNIST Dataset from disk...")
        mnist = joblib.load('./data/mnist_dataset.joblib')
    else:
        ui.announce("Downloading MNIST Dataset using openml...")
        mnist = fetch_openml(name='mnist_784', version=1)
        joblib.dump(mnist, './data/mnist_dataset.joblib')
    return mnist


def load_and_preprocess_image(image_path, input_shape):
    image = Image.open(image_path)
    # Resize the image to match the input shape of your models

    # Convert the PIL image to a numpy array
    image_array = np.array(image)

    # Flatten the image array
    image_flattened = image_array.flatten()

    # Ensure the shape is (784,)
    image_flattened = image_flattened.reshape(-1)

    # Ensure the length is 784
    image_flattened = image_flattened[:784]

    return image_flattened


# def load_and_preprocess_image(image_path, input_shape):
#     image = Image.open(image_path)
#     # Resize the image to match the input shape of your models
#     ...


class MachineLearning:
    def __init__(self, name, input_shape):
        self.name = name
        self.model = None
        self.accuracy = None
        self.input_shape = input_shape

    def save(self):
        ui.announce("Saving Model & Accuracy...")

        # Save the trained model and accuracy using joblib
        joblib.dump(self.model, f'models/{self.name}_model.joblib')
        with open(f'models/{self.name}_accuracy.txt', 'w') as f:
            f.write(str(self.accuracy))

    def load_model(self):
        ui.announce("Loading Model...")

        self.model = joblib.load(f'models/{self.name}_model.joblib')

    def get_predictions_for_folder(self):
        folder = 'data/my_data'
        predictions = []
        image_files = os.listdir(folder)
        for image_file in image_files:
            actual = int(image_file.split('_')[0])
            image_path = os.path.join(folder, image_file)
            image = load_and_preprocess_image(image_path, self.input_shape)
            prediction = self.model.predict(np.expand_dims(image, axis=0))
            predictions.append(
                {'file': image_file,
                 'actual': actual,
                 'predictions': prediction,
                 'predicted': np.argmax(prediction)})
        return predictions

    def __str__(self):
        return "MachineLearning(name=%s)" % self.name
