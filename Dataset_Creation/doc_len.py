import os


gold_full_path = "../data/sub_sample/gold_full_20/"

for file in os.listdir(gold_full_path):
    file_path = gold_full_path + file
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
        tokenized_text = file_content.split(' ')
        print(file, len(tokenized_text))
