import json
import os.path
import random
from Dataset_Creation.TextManipulation.CoherenceManipulator import CoherenceManipulator
from Dataset_Creation.TextManipulation.FluencyManipulator import FluencyManipulator


def apply_manipulation(elem, dataset, params_c, params_f, affected_metric, category, manipulation_type, range_type,
                       origin, subset, subset_num, exp_number):
    """
    Apply a text manipulation to a gold-standard document for dataset augmentation.

    This function modifies the input text using coherence- or fluency-based manipulations.
    It saves the manipulated text and returns a metadata entry for further processing.

    Parameters:
        elem (dict): Metadata entry for the original document.
        dataset: Full list of metadata entries for the dataset.
        params_c (dict): Parameters for coherence manipulations.
        params_f (dict): Parameters for fluency manipulations.
        affected_metric (str): The metric targeted ("coherence" or "fluency").
        category (str): Sub-category of manipulation (e.g., "anaphora_resolution").
        manipulation_type (str): Specific manipulation to apply.
        range_type (str): Manipulation range ("document", "paragraph", etc.).
        origin (str): Path to the dataset's base directory.
        subset (str): Name of the dataset subset (e.g., "pg-1900").
        subset_num (str): Subset scope (e.g., "full").
        exp_number (int): Identifier for the experiment run.

    Returns:
        dict or None: A dictionary containing metadata for the manipulated sample,
                      or None if the manipulation failed (e.g., no text returned).

    Raises:
        None explicitly, but may propagate file or processing errors.
    """
    idx = elem['id']
    with open(f"../{elem['gold_text']}", 'r', encoding='utf-8') as in_file:
        sample_gold_text = in_file.read()

    if affected_metric == "coherence":  # COHERENCE
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
        elif manipulation_type == "exchange_content":
            insert_texts = list()
            if not params_c['ratio']:
                # randomly choose 'texts_to_insert' in dataset unequal to current text based on 'n'
                sample_indices = random.sample(range(len(dataset)), k=params_c['n'])
                for sample_idx in sample_indices:
                    while dataset[sample_idx]['id'] == idx:
                        sample_idx = random.choice(range(len(dataset)))

                    sample_insert_path = dataset[sample_idx]['gold_text']
                    with open(f"..{sample_insert_path}", 'r', encoding='utf-8') as in_file:
                        sample_gold_text_insert = in_file.read()
                        insert_texts.append(sample_gold_text_insert)
            else:
                # load all available documents as it is unknown beforehand, how many documents will be needed as 'texts_to_insert'
                sample_indices = [i for i in range(len(dataset)) if dataset[i]['id'] != idx]

                for sample_idx in sample_indices:
                    sample_insert_path = dataset[sample_idx]['gold_text']
                    with open(f"..{sample_insert_path}", 'r', encoding='utf-8') as in_file:
                        sample_gold_text_insert = in_file.read()
                        insert_texts.append(sample_gold_text_insert)

            params_c['texts_to_insert'] = insert_texts

            manipulated_text, manipulated_char_spans = manipulator.apply_manipulation(
                manipulation_type=manipulation_type,
                category=category,
                range_type=range_type,
                n=params_c['n'],
                min_len=params_c['min_len'],
                texts_to_insert=insert_texts,
                ratio=params_c['ratio']
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
        elif manipulation_type == "exchange_entities_w_pronouns":
            manipulated_text, manipulated_char_spans = manipulator.apply_manipulation(
                manipulation_type=manipulation_type,
                category=category,
                entities_to_consider=params_c['entities_to_consider'],
                start=params_c['start'],
                end=params_c['end'],
            )
        elif manipulation_type == "exchange_entities_w_term":
            manipulated_text, manipulated_char_spans = manipulator.apply_manipulation(
                manipulation_type=manipulation_type,
                category=category,
                entities_to_consider=params_c['entities_to_consider'],
                start=params_c['start'],
                end=params_c['end'],
                replacement_terms=params_c['replacement_terms']
            )
        elif manipulation_type == "exchange_entities" or manipulation_type == "swap_entities":
            manipulated_text, manipulated_char_spans = manipulator.apply_manipulation(
                manipulation_type=manipulation_type,
                category=category,
                start=params_c['start'],
                end=params_c['end'],
                entity=params_c['entity']
            )
        else:
            manipulated_text, manipulated_char_spans = manipulator.apply_manipulation(
                manipulation_type=manipulation_type,
                category=category,
                range_type=range_type,
                n=params_c['n'],
                min_len=params_c['min_len']
            )
    else:  # FLUENCY
        if (manipulation_type == "typos" or
                manipulation_type == "verb_tenses" or
                manipulation_type == "punctuation_removal" or
                manipulation_type == "word_order"):
            # create coherence manipulator
            manipulator = FluencyManipulator(sample_gold_text, idx)
            manipulated_text, manipulated_char_spans = manipulator.apply_manipulation(
                manipulation_type=manipulation_type,
                category=category,
                start=params_f['start'],
                end=params_f['end'],
                dense=params_f['dense']
            )
            range_type = 'document'
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
    mani_idx = f"{idx}_{ID_MAPPING[affected_metric]}_{ID_MAPPING[category]}_{ID_MAPPING[manipulation_type]}"
    if affected_metric == "coherence":
        out_path_dir = f"{origin}/manipulated_{subset}_{subset_num}/{ID_MAPPING[affected_metric]}/{ID_MAPPING[category]}/{ID_MAPPING[manipulation_type]}/{range_type}/exp_{exp_number}/"
    else:
        out_path_dir = f"{origin}/manipulated_{subset}_{subset_num}/{ID_MAPPING[affected_metric]}/{ID_MAPPING[category]}/{ID_MAPPING[manipulation_type]}/exp_{exp_number}/"

    out_path = out_path_dir + f"{idx}.txt"
    if not os.path.isdir(out_path_dir):
        os.makedirs(out_path_dir)

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

ID_MAPPING = json.load(open("./templates/id_mapping.json", 'r', encoding='utf-8'))

subset = "pg-1900"
subset_num = "full"
origin = "../data/project_gutenberg"
gold_data_path = f"{origin}/gold_{subset}_meta_{subset_num}.json"
manipulation_types_path = "./TextManipulation/manipulation_types.json"

METRIC = "coherence"
CATEGORY = "anaphora_resolution"
MANIPULATION = "exchange_entities_w_term"
RANGE = "document"
EXP_NUMBER = 1

# --PARAMS--
params_coherence = {
    'n': 0,  # number of operations to perform
    'min_len': 50,  # minimum number of characters a paragraph needs to have to be considered as a paragraph
    'repetition_factor': None,  # repeat_content
    'start': 35,
    'end': 35,
    'ratio': False,  # exchange_content
    'entity': None,  # swap_entities, exchange_entities
    'entities_to_consider': ["LOC", "GPE", "FAC", "ORG"],  # exchange_entities_w_pronouns, exchange_entities_w_term
    'replacement_terms': ['stuff', 'thing']  # exchange_entities_w_term
}

params_fluency = {
    'start': 5,  # start value (range): at document-level
    'end': 5,  # end value (range): at document-level
    'dense': False
}

# process all gold documents
with open(gold_data_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # meta data for PG dataset

    entries_coherence = list()
    entries_fluency = list()

    for dp in data:  # limit size for testing
        print(dp['id'])
        # entry = apply_manipulation_random(dp, data, params_coherence, params_fluency)
        entry = apply_manipulation(
            elem=dp,
            dataset=data,
            params_c=params_coherence,
            params_f=params_fluency,
            affected_metric=METRIC,
            category=CATEGORY,
            manipulation_type=MANIPULATION,
            range_type=RANGE,
            origin=origin,
            subset=subset,
            subset_num=subset_num,
            exp_number=EXP_NUMBER
        )

        if entry:  # save meta data based on affected metric
            if entry['affected_metric'] == 'coherence':
                entries_coherence.append(entry)
            else:
                entries_fluency.append(entry)

    # save meta data for entries
    if entries_fluency:
        # FLUENCY
        out_path_fluency = f"{origin}/manipulated_{subset}_{subset_num}/{ID_MAPPING[METRIC]}/{ID_MAPPING[CATEGORY]}/{ID_MAPPING[MANIPULATION]}/exp_{EXP_NUMBER}_meta.json"
        with open(out_path_fluency, 'w', encoding='utf-8') as out_file:
            json.dump(entries_fluency, out_file, indent=4)

    if entries_coherence:
        # COHERENCE
        out_path_coherence = f"{origin}/manipulated_{subset}_{subset_num}/{ID_MAPPING[METRIC]}/{ID_MAPPING[CATEGORY]}/{ID_MAPPING[MANIPULATION]}/{RANGE}/exp_{EXP_NUMBER}_meta.json"
        with open(out_path_coherence, 'w', encoding='utf-8') as out_file:
            json.dump(entries_coherence, out_file, indent=4)
