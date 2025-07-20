from transformers import AutoTokenizer
import json
import os


def load_tokenizer(tokenizer_name):
    """
    Loads a tokenizer from Hugging Face's Transformers library.

    Args:
        tokenizer_name (str): Name or path of the pretrained tokenizer to load.

    Returns:
        transformers.PreTrainedTokenizer: The loaded tokenizer object.
    """
    return AutoTokenizer.from_pretrained(tokenizer_name)


def load_prompt_template(prompt_path):
    """
    Loads a prompt template from a text file.

    Args:
        prompt_path (str): Path to the prompt template file.

    Returns:
        str: The content of the prompt template.
    """
    with open(prompt_path, 'r', encoding='utf-8') as prompt_file:
        return prompt_file.read()


def get_prompt_1(text, sys_prompt, prompt_template):
    """
    Creates a conversation prompt with one variable (`Story`).

    Args:
        text (str): The text to be inserted as `Story`.
        sys_prompt (str): The system-level prompt.
        prompt_template (str): Template containing a `{Story}` placeholder.

    Returns:
        list[dict]: A list of dicts representing a conversation.
    """
    conversation = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": prompt_template.format(Story=text)
        }
    ]
    return conversation


def get_prompt_2(content, add_on, sys_prompt, prompt_template):
    """
    Creates a conversation prompt with two variables (`Content`, `AddOn`).

    Args:
        content (str): Main content input.
        add_on (str): Additional context or input.
        sys_prompt (str): The system-level prompt.
        prompt_template (str): Template with `{Content}` and `{AddOn}` placeholders.

    Returns:
        list[dict]: A list of dicts representing a conversation.
    """
    conversation = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": prompt_template.format(Content=content, AddOn=add_on)
        }
    ]
    return conversation


def get_prompt_3(text, content, add_on, sys_prompt, prompt_template):
    """
    Creates a conversation prompt with three variables (`Content`, `AddOn`, `Text`).

    Args:
        text (str): Main text input.
        content (str): Supplementary content.
        add_on (str): Additional context or instructions.
        sys_prompt (str): The system-level prompt.
        prompt_template (str): Template with `{Text}`, `{Content}`, and `{AddOn}` placeholders.

    Returns:
        list[dict]: A list of dicts representing a conversation.
    """
    conversation = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": prompt_template.format(Content=content, AddOn=add_on, Text=text)
        }
    ]
    return conversation


def load_meta_data(path):
    """
    Loads metadata from a JSON file.

    Args:
        path (str): Path to the metadata file.

    Returns:
        dict: Parsed JSON content as a dictionary.
    """
    with open(path, 'r', encoding='utf-8') as data:
        return json.load(data)


def read_file(filepath: str) -> str:
    """
    Reads the contents of a file and returns it as a string.

    Args:
        filepath (str): Path to the file.

    Returns:
        str: Contents of the file.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


def save_to_json(data: dict, output_path: str):
    """
    Saves a dictionary as a JSON file.

    Args:
        data (dict): Data to save.
        output_path (str): Path to save the JSON file to.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def save_to_text(report: str, output_path: str):
    """
    Saves a string to a text file.

    Args:
        report (str): Text content to write.
        output_path (str): Path to save the text file to.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(report)
