from transformers import AutoTokenizer
import json
import os


def load_tokenizer(tokenizer_name):
    return AutoTokenizer.from_pretrained(tokenizer_name)


def load_prompt_template(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as prompt_file:
        return prompt_file.read()


def get_prompt_1(text, sys_prompt, prompt_template):
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
    with open(path, 'r', encoding='utf-8') as data:
        return json.load(data)


def read_file(filepath: str) -> str:
    """
    Reads the contents of a file and returns it as a string.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


def save_to_json(data: dict, output_path: str):
    """
    Saves the parsed data dictionary to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def save_to_text(report: str, output_path: str):
    """
    Saves the created report to a .txt file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(report)
