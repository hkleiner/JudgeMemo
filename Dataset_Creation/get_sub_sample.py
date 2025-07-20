import os
import random
import json


"""
Sub-sample Project Gutenberg dataset files for full and partial text storage.

This script loads metadata for Project Gutenberg texts, filters the texts 
based on chapter presence or metadata inclusion, and extracts either full or 
partial (first N tokens) versions of the texts. It saves a random selection 
of these texts into specified directories.

Configuration:
    SEED (int): Random seed for reproducibility.
    SUBSET_NUM (int): Number of tokens to extract for partial text.
    SELECT_N (int): Number of files to randomly select and save.
    CHAPTER_ONLY (bool): If True, only files containing 'CHAPTER: ' and
                         with token length >= 8000 are included.

Paths:
    pg_path (str): Directory containing Project Gutenberg text files.
    meta_path (str): JSON metadata file path for Project Gutenberg dataset.
    gold_full_path (str): Directory to save the full text files.
    gold_sanity_path (str): Directory to save the partial text files.
"""


SEED = 42
SUBSET_NUM = 2000
SELECT_N = 65
CHAPTER_ONLY = False
file_section = {}

pg_path = "../data/project_gutenberg/gold_dataset/"
meta_path = "../data/project_gutenberg/gold_pg-1900_meta_full.json"
gold_full_path = "../data/sub_sample/gold_full_20/"
gold_sanity_path = "../data/project_gutenberg/gold_dataset_2000/"

with open(meta_path, 'r', encoding='utf-8') as data:
    meta = json.load(data)
    ids = []

    for entry in meta:
        idx = 'PG' + entry['id'][2:] + '.txt'
        ids.append(idx)

for file in os.listdir(pg_path):
    file_path = pg_path + file
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

        if CHAPTER_ONLY:
            if "CHAPTER: " in file_content:
                tokenized_text = file_content.split(' ')
                len_text = len(tokenized_text)
                if len_text >= 8000:
                    file_section[file] = [file_content, ' '.join(tokenized_text[:SUBSET_NUM])]  # [full_text, first 1000 tokens]
        else:
            if file in ids:
                tokenized_text = file_content.split(' ')
                len_text = len(tokenized_text)
                file_section[file] = [file_content, ' '.join(tokenized_text[:SUBSET_NUM])]

print(len(file_section))
selection = random.sample(list(file_section.items()), k=SELECT_N)
print(len(selection))

i = 1
for (filename, texts) in selection:
    full, part = texts[0], texts[1]
    with open(gold_full_path+filename, 'w', encoding='utf-8') as full_file:
        full_file.write(full)

    with open(gold_sanity_path+filename, 'w', encoding='utf-8') as sanity_file:
        sanity_file.write(part)

    print(f"Saved {i}, {filename}")
    i += 1

