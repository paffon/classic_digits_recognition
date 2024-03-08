import time

import ui.ui as ui
import core.module_cnn as module_cnn
import core.module_knn as module_knn
import core.utils as my_utils


def main():

    # Ask the user to choose between training live,
    # or loading a pre-trained model
    train = 'Train a brand new model. Exciting!'
    load = 'Load a pre-trained model and perform predictions!'
    train_or_load = ui.ask_user(
        prompt="Do you want to train a new model, or load a"
               " pre-trained one?",
        options=[train,
                 load]
    )

    # Ask the user which machine learning type they'd like to use
    chosen_model_name = ui.ask_user(
        prompt=f"Awesome! Which model type do"
               f" you want to {train_or_load.split(' ')[0].lower()}?",
        options=[module_cnn.class_name,
                 module_knn.class_name]
    )

    # Determine the machine learning class to be used
    learner_class = {
        module_cnn.class_name: module_cnn.NN,
        module_knn.class_name: module_knn.KNN,
    }[chosen_model_name]

    # Get trained model (Train live or load pre-trained)
    instance = learner_class()
    if train_or_load == train:
        instance.train_and_save()
    elif train_or_load == load:
        instance.load_model()
    else:
        raise ValueError(f"Invalid action {train_or_load}")

    # Ask the user which test set they'd like to use
    chosen_test_set = ui.ask_user(
        prompt=f"Which test set do you want to use?",
        options=['my_samples', 'mnist_samples'])

    # Use the trained model of the chosen machine learning class to make some
    # predictions on the chosen test set
    predictions = instance.get_predictions_for_folder(chosen_test_set)

    # Display the results of the predictions
    print('Incorrect:')
    for incorrect_prediction in predictions['incorrect']:
        print(f'\t{incorrect_prediction}')

    predictions_accuracy = len(predictions['correct']) / (len(predictions['correct']) + len(predictions['incorrect']))
    print(f'Test accuracy is:  {round(predictions_accuracy, 4)}\n'
          f'Model accuracy is: {round(instance.accuracy, 4)}')


if __name__ == "__main__":
    main()
