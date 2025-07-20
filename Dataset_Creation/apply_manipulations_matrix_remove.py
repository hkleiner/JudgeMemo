import json
import os.path
import random
from Dataset_Creation.TextManipulation.CoherenceManipulator import CoherenceManipulator


GOLD_PATH = "../data/sub_sample/gold_full_20/"


def apply_manipulation_removal_filler(
        elem,
        params_c,
        origin,
        subset,
        subset_num,
        exp_number,
        affected_metric="coherence",
        category="logical_flow_disruptions",
        manipulation_type="remove_content",
        range_type="paragraph"
):
    idx = elem['id']
    with open(f"../{elem['gold_text']}", 'r', encoding='utf-8') as in_file:
        sample_gold_text = in_file.read()

    manipulator = CoherenceManipulator(sample_gold_text, idx)
    manipulated_text, manipulated_char_spans = manipulator.apply_manipulation(
        manipulation_type=manipulation_type,
        category=category,
        range_type=range_type,
        n=params_c['n'],
        min_len=params_c['min_len']
    )

    # If manipulation fails, manipulated_text will be None
    if not manipulated_text:
        return None

    # save manipulated_text
    mani_idx = f"{idx}_{ID_MAPPING[affected_metric]}_{ID_MAPPING[category]}_{ID_MAPPING[manipulation_type]}"
    out_path_dir = f"{origin}/manipulated_{subset}_{subset_num}/{ID_MAPPING[affected_metric]}/{ID_MAPPING[category]}/{ID_MAPPING[manipulation_type]}/{range_type}/exp_{exp_number}/"

    out_path = out_path_dir + f"{idx}.txt"
    if not os.path.isdir(out_path_dir):
        os.makedirs(out_path_dir)

    # as content gets removed, add tokens from gold document in the end, so that we habe subset_num tokens again
    tokenized_manipulated_text = manipulated_text.split(' ')
    num_mani_tokens = len(tokenized_manipulated_text)
    missing_tokens = subset_num - num_mani_tokens
    print(f"current length: {num_mani_tokens}; missing to {subset_num}: {missing_tokens}")

    file_path = GOLD_PATH + "PG-" + elem['gold_source']['source_id'] + ".txt"
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
        tokenized_gold_text = file_content.split(' ')
        print(f"end span: {subset_num+missing_tokens}")
        continuation = tokenized_gold_text[subset_num:subset_num+missing_tokens]
        tokenized_manipulated_text += continuation
        print(f"new length: {len(tokenized_manipulated_text)}")

    manipulated_text = ' '.join(tokenized_manipulated_text)

    with open(out_path, 'w', encoding='utf-8') as out_file:
        out_file.write(manipulated_text)

    entry = {
        "id": mani_idx,
        "manipulated_text": out_path[2:],
        "manipulated_char_spans": manipulated_char_spans,
        "range_type": range_type,
        "affected_metric": affected_metric,
        "manipulation_category": category,
        "manipulation_type": manipulation_type,
        "gold_id": idx,
        "gold_text": elem['gold_text'],
        "random_seed": SEED,
        "params": params_c
    }

    return entry


# ------CONFIG------
SEED = 31
random.seed(SEED)

ID_MAPPING = json.load(open("./templates/id_mapping.json", 'r', encoding='utf-8'))

subset = "sanity"
subset_num = 1000
origin = "../data/sub_sample"
gold_data_path = f"{origin}/gold_{subset}_meta_{subset_num}.json"
manipulation_types_path = "./TextManipulation/manipulation_types.json"

EXP_NUMBER = 4

# --PARAMS--
params_coherence = {
    'n': 4,  # number of operations to perform
    'min_len': 50,  # minimum number of characters a paragraph needs to have to be considered as a paragraph
}

# process all gold documents
with open(gold_data_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # meta data for PG dataset

    entries_coherence = list()

    for dp in data:  # limit size for testing
        # entry = apply_manipulation_random(dp, data, params_coherence, params_fluency)
        entry = apply_manipulation_removal_filler(
            elem=dp,
            params_c=params_coherence,
            origin=origin,
            subset=subset,
            subset_num=subset_num,
            exp_number=EXP_NUMBER
        )

        entries_coherence.append(entry)

    # save meta data for entries
    # COHERENCE
    out_path_coherence = f"{origin}/manipulated_{subset}_{subset_num}/{ID_MAPPING['coherence']}/{ID_MAPPING['logical_flow_disruptions']}/{ID_MAPPING['remove_content']}/paragraph/exp_{EXP_NUMBER}_meta.json"
    with open(out_path_coherence, 'w', encoding='utf-8') as out_file:
        json.dump(entries_coherence, out_file, indent=4)
