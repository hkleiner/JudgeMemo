import os
import json


"""
Create shortened 'sanity check' dataset from full Project Gutenberg samples.

Reads full text files and their metadata, truncates each text to the first
3000 tokens, saves the shortened texts in a new directory, and updates the 
metadata to point to these shortened texts.

Paths:
    gold_full_path (str): Directory containing full text files.
    gold_full_meta (str): JSON metadata file for full text files.
    out_path (str): Directory to save shortened text files.
    meta_out_path (str): Path to save updated metadata JSON.
"""


gold_full_path = "../data/sub_sample/gold_full_20/"
gold_full_meta = "../data/sub_sample/gold_full_meta_20.json"
out_path = "../data/sub_sample/gold_sanity_3000/"
meta_out_path = "../data/sub_sample/gold_sanity_meta_3000.json"

# Load the full metadata JSON and update the 'gold_text' paths to point to shortened files
with open(gold_full_meta, 'r', encoding='utf-8') as meta:
    meta_content = json.load(meta)

    for entry in meta_content:
        # Adjust 'gold_text' path to new shortened file location, trimming original path accordingly
        entry["gold_text"] = out_path[2:] + entry["gold_text"].split('/')[4]

# Save updated metadata JSON for shortened texts
with open(meta_out_path, 'w', encoding='utf-8') as o_meta:
    o_meta.write(json.dumps(meta_content, indent=4))

# Iterate through full text files, truncate to first 3000 tokens, and save to new directory
for f in os.listdir(gold_full_path):
    filename = gold_full_path + f
    with open(filename, 'r', encoding='utf-8') as in_file:
        content = in_file.read()
        tokenized_text = content.split(' ')
        shorted_content = ' '.join(tokenized_text[:3000])

    with open(out_path+f, 'w', encoding='utf-8') as out:
        out.write(shorted_content)

