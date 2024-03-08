import os

import joblib
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml

import ui.ui as ui
import core.utils as my_utils


def fetch_mnist_data():
    if os.path.exists(f'./data/mnist_dataset.joblib'):
        ui.announce("Loading MNIST Dataset from disk...")
        mnist = joblib.load('./data/mnist_dataset.joblib')
    else:
        ui.announce("Downloading MNIST Dataset using openml...")
        mnist = fetch_openml(name='mnist_784', version=1)
        joblib.dump(mnist, './data/mnist_dataset.joblib')
    return mnist


class MachineLearning:
    def __init__(self, name, input_shape):
        self.name = name
        self.model = None
        self.accuracy = None
        self.input_shape = input_shape

        try:
            with open(f'models/{self.name}_accuracy.txt') as file:
                self.accuracy = float(file.read())
        except FileNotFoundError:
            print('No accuracy found. If you want to have accuracy, train first.')

    def save(self):
        ui.announce("Saving Model & Accuracy...")

        # Save the trained model and accuracy using joblib
        joblib.dump(self.model, f'models/{self.name}_model.joblib')
        with open(f'models/{self.name}_accuracy.txt', 'w') as f:
            f.write(str(self.accuracy))

    def train_and_save(self):
        ui.announce("Training!")

        params = my_utils.read_hyperparameters(self.name)
        self.train(params)
        self.save()

    def load_model(self):
        ui.announce("Loading Model...")

        self.model = joblib.load(f'models/{self.name}_model.joblib')

    def get_predictions_for_folder(self, chosen_test_set):
        folder = f'data/{chosen_test_set}'
        predictions = {'correct': [], 'incorrect': []}
        image_files = os.listdir(folder)
        for image_file in image_files:
            actual = int(image_file.split('_')[0])

            # Open the image
            image = Image.open(os.path.join(folder, image_file))

            prediction, correct_prediction = self.predict(image, actual)

            element = {'file': image_file,
                       'actual': actual,
                       'predicted': np.argmax(prediction)}

            if correct_prediction:
                predictions['correct'].append(element)
            else:
                predictions['incorrect'].append(element)

        return predictions

    def __str__(self):
        return "MachineLearning(name=%s)" % self.name

    def train(self, params):
        raise NotImplementedError("train() should be implemented by the"
                                  "specific machine learning class")

    def predict(self, image, actual):
        raise NotImplementedError("predict() should be implemented by the"
                                  "specific machine learning class")
