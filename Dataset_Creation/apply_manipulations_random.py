import json
import os.path
import random
from Dataset_Creation.TextManipulation.CoherenceManipulator import CoherenceManipulator
from Dataset_Creation.TextManipulation.FluencyManipulator import FluencyManipulator


manipulation_types_path = "./TextManipulation/manipulation_types.json"
ID_MAPPING = json.load(open("./templates/id_mapping.json", 'r', encoding='utf-8'))
MANIPULATION_TYPES = json.load(open(manipulation_types_path, 'r', encoding='utf-8'))
RANGE_TYPE = ["paragraph", "section", "chapter"]
EXP_NUMBER = 1
METRICS = MANIPULATION_TYPES.keys()


def apply_manipulation_random(elem, dataset, params_c, params_f, origin):
    idx = elem['id']
    with open(f"../{elem['gold_text']}", 'r', encoding='utf-8') as in_file:
        sample_gold_text = in_file.read()

    # randomly set config values for manipulation
    affected_metric = random.choice(list(METRICS))
    category = random.choice(list(MANIPULATION_TYPES[affected_metric].keys()))
    manipulation_type = random.choice(MANIPULATION_TYPES[affected_metric][category])
    range_type = random.choice(RANGE_TYPE) if affected_metric == "coherence" else "document"

    if affected_metric == "coherence":
        # create coherence manipulator
        manipulator = CoherenceManipulator(sample_gold_text, idx)
        if manipulation_type == "insert_content":
            # randomly choose text in dataset unequal to current text
            sample_idx = random.choice(range(len(dataset)))
            while dataset[sample_idx]['id'] == idx:
                sample_idx = random.choice(range(len(dataset)))

            sample_insert_path = dataset[sample_idx]['gold_text']
            with open(f"..{sample_insert_path}", 'r', encoding='utf-8') as in_file:
                sample_gold_text_insert = in_file.read()

            params_c['text_to_insert'] = sample_insert_path

            manipulated_text, manipulated_char_spans = manipulator.apply_manipulation(
                manipulation_type=manipulation_type,
                category=category,
                range_type=range_type,
                n=params_c['n'],
                min_len=params_c['min_len'],
                text_to_insert=sample_gold_text_insert
            )
        elif manipulation_type == "repeat_content":
            manipulated_text, manipulated_char_spans = manipulator.apply_manipulation(
                manipulation_type=manipulation_type,
                category=category,
                range_type=range_type,
                n=params_c['n'],
                min_len=params_c['min_len'],
                repetition_factor=params_c['repetition_factor']
            )
        else:
            manipulated_text, manipulated_char_spans = manipulator.apply_manipulation(
                manipulation_type=manipulation_type,
                category=category,
                range_type=range_type,
                n=params_c['n'],
                min_len=params_c['min_len']
            )
    else:
        # create coherence manipulator
        manipulator = FluencyManipulator(sample_gold_text, idx)
        manipulated_text, manipulated_char_spans = manipulator.apply_manipulation(
            manipulation_type=manipulation_type,
            category=category,
            start=params_f['start'],
            end=params_f['end']
        )
        range_type = 'document'

    # If manipulation fails, manipulated_text will be None
    if not manipulated_text:
        return None

    # save manipulated_text
    mani_idx = f"{idx}_{ID_MAPPING[affected_metric]}_{ID_MAPPING[category]}_{ID_MAPPING[manipulation_type]}_{range_type}"
    out_path = f"{origin}/manipulated_datasets/project_gutenberg/exp_{EXP_NUMBER}/{mani_idx}.txt"
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
        "params": params_c if affected_metric == 'coherence' else params_f
    }
    return entry


# ------CONFIG------
SEED = 31
random.seed(SEED)

subset = "sanity"
origin = "../data/gold_datasets"
gold_data_path = f"{origin}/final_pg_gold_meta.json"

# --PARAMS--
params_coherence = {
    'n': 3,  # number of operations to perform
    'min_len': 50,  # minimum number of characters a paragraph needs to have to be considered as a paragraph
    'repetition_factor': 3
}

params_fluency = {
    'start': 100,  # start value (range): at document-level
    'end': 100,  # end value (range): at document-level
}

# process all gold documents
with open(gold_data_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # meta data for PG dataset

    entries_coherence = list()
    entries_fluency = list()

    for dp in data:  # limit size for testing
        # entry = apply_manipulation_random(dp, data, params_coherence, params_fluency)
        entry = apply_manipulation_random(
            elem=dp,
            dataset=data,
            params_c=params_coherence,
            params_f=params_fluency,
            origin=origin
        )

        if entry:  # save meta data based on affected metric
            if entry['affected_metric'] == 'coherence':
                entries_coherence.append(entry)
            else:
                entries_fluency.append(entry)

    # save meta data for entries
    if entries_fluency:
        # FLUENCY
        out_path_fluency = f"{origin}/manipulated_datasets/project_gutenberg/exp_{EXP_NUMBER}_meta.json"
        with open(out_path_fluency, 'w', encoding='utf-8') as out_file:
            json.dump(entries_fluency, out_file, indent=4)

    if entries_coherence:
        # COHERENCE
        out_path_coherence = f"{origin}/manipulated_datasets/project_gutenberg/exp_{EXP_NUMBER}_meta.json"
        with open(out_path_coherence, 'w', encoding='utf-8') as out_file:
            json.dump(entries_coherence, out_file, indent=4)
