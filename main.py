import time

import ui.ui as ui
from core.module_nn import NN
from core.module_knn import KNN
import core.utils as my_utils


def main():

    train_or_load = ui.ask_user(
        prompt="Do you want to train a new model, or load a"
               " pre-trained one?",
        options=["Train", "Load"]
    )

    nn = "nn"
    knn = "knn"
    chosen_model_name = ui.ask_user(
        prompt=f"Awesome! Let's {train_or_load.lower()}. Which model type do"
               f" you want to {train_or_load.lower()}?",
        options=[nn, knn]
    )

    learner_class = {
        nn: NN,
        knn: KNN,
    }[chosen_model_name]

    instance = learner_class()

    if train_or_load == "Train":

        ui.announce("Training!")

        params = my_utils.read_hyperparameters(chosen_model_name)
        instance.train(params)
        instance.save()
    elif train_or_load == "Load":
        instance.load_model()
    else:
        raise ValueError(f"Invalid action {train_or_load}")

    predict_or_report = ui.ask_user(
        prompt="Do you want to try the model or produce a report?",
        options=["Predict", "Report"]
    )

    if predict_or_report == "Predict":
        predictions = instance.get_predictions_for_folder()
    elif predict_or_report == "Report":
        instance.report()
    else:
        raise ValueError(f"Invalid action {predict_or_report}")










if __name__ == "__main__":
    main()
