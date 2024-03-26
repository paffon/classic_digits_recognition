from typing import List, Any, Union


def separate() -> None:
    """
    Print a separator line to visually separate sections.

    :return: None
    """
    separator = "=" * 20
    print(f"\n{separator}\n")


def announce(message: str) -> None:
    """
    Print a message with a separator above and below it.

    :param message: The message to be printed.
    :return: None
    """
    separate()
    print(message)


def ask_user(prompt: str, options: Union[None, List[Any]]) -> Any:
    """
    Prompt the user with a message and a list of options, then return the user's choice.

    :param prompt: The prompt to be displayed to the user.
    :param options: A list of options from which the user can choose.
    :return: The user's choice.
    """
    announce(prompt)
    for i, option in enumerate(options, start=1):
        print(f"[{i}] {option}")

    user_input = input(f"Enter your choice (1 - {len(options)}): ")

    user_choice = options[int(user_input) - 1]

    return user_choice


def progress_bar(current: int, total: int, width: int = 80, full: str = '#', empty: str = '_') -> str:
    """
    Generate a progress bar string representing the progress of a task.

    :param current: The current progress.
    :param total: The total progress.
    :param width: The width of the progress bar.
    :param full: The character representing completed progress.
    :param empty: The character representing remaining progress.
    :return: A string representing the progress bar.
    """
    progress = width * current // total
    progress_string = '[' + full * progress + empty * (width - progress) + ']'

    return progress_string


def ask_user_questions(train: str, load: str, module_cnn_name: str, module_knn_name: str) -> tuple[str, str]:
    """
    Function to ask the user questions related to model training and selection.

    :param train: Description of the option to train a new model
    :param load: Description of the option to load a pre-trained model
    :param module_cnn_name: Name of the CNN module
    :param module_knn_name: Name of the KNN module
    :return: Tuple containing the user's choice for training or loading and the chosen model name
    """
    train_or_load = ask_user(
        prompt="Do you want to train a new model, or load a pre-trained one?",
        options=[train, load]
    )

    # Ask the user which machine learning type they'd like to use
    chosen_model_name = ask_user(
        prompt=f"Awesome! Which model type do you want to {train_or_load.split(' ')[0].lower()}?",
        options=[module_cnn_name, module_knn_name]
    )

    return train_or_load, chosen_model_name

