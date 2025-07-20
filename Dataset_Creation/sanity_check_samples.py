import os
import json


gold_full_path = "../data/sub_sample/gold_full_20/"
gold_full_meta = "../data/sub_sample/gold_full_meta_20.json"
out_path = "../data/sub_sample/gold_sanity_3000/"
meta_out_path = "../data/sub_sample/gold_sanity_meta_3000.json"

with open(gold_full_meta, 'r', encoding='utf-8') as meta:
    meta_content = json.load(meta)

    for entry in meta_content:
        entry["gold_text"] = out_path[2:] + entry["gold_text"].split('/')[4]

with open(meta_out_path, 'w', encoding='utf-8') as o_meta:
    o_meta.write(json.dumps(meta_content, indent=4))

for f in os.listdir(gold_full_path):
    filename = gold_full_path + f
    with open(filename, 'r', encoding='utf-8') as in_file:
        content = in_file.read()
        tokenized_text = content.split(' ')
        shorted_content = ' '.join(tokenized_text[:3000])

    with open(out_path+f, 'w', encoding='utf-8') as out:
        out.write(shorted_content)

