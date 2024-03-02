from typing import List, Any, Union


def separate():
    separator = "=" * 20
    print(f"\n{separator}\n")


def announce(message: str):
    separate()
    print(message)

def ask_user(prompt: str, options: Union[None, List[Any]]) -> Any:

    announce(prompt)
    for i, option in enumerate(options, start=1):
        print(f"[{i}] {option}")

    user_input = input(f"Enter your choice (1 - {len(options)}): ")

    user_choice = options[int(user_input) - 1]

    return user_choice


def progress_bar(current, total, width=80, full='#', empty='_'):
    progress = width * current // total
    progress_string = '[' + full * progress + empty * (width - progress) + ']'

    return progress_string
