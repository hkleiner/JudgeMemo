import os
import json
import DatasetCreator.PGPreprocessor as pgp
from DatasetCreator.utils import save_metadata, get_formatted_idx


def run_step_1_2(path, story_8_16_path, out_dir):
    with (open(path, 'r', encoding='utf-8') as file,
          open(story_8_16_path, 'r', encoding='utf-8') as f_story):
        stories = json.load(f_story)
        cleaned_gutenberg = json.load(file)

        # SEMI-AUTOMATED CLEAN-UP OF DOCUMENTS
        # ignore anything that indicates not being epic work (poems, songs, lyric, ...)
        for sample in cleaned_gutenberg:
            for (idx, _) in stories:
                pg_id = sample['id']
                story = sample['text']

                if idx == pg_id:
                    file_path = f"{out_dir}PG-{get_formatted_idx(pg_id)}.txt"
                    story_beginning = story[:250].lower()
                    if (  # add other keywords as you wish
                            'poem' in story_beginning or
                            'song' in story_beginning or
                            'rhym' in story_beginning or
                            'prose' in story_beginning or
                            'tragedy' in story_beginning or
                            'drama' in story_beginning or
                            'scene' in story_beginning or
                            'act' in story_beginning or
                            'lyric' in story_beginning
                    ):
                        continue
                    else:
                        with open(file_path, 'w', encoding='utf-8') as out:
                            out.write(story)


def run_step_3(out_dir, path, final_gold_meta_path_pg):
    pg_dataset_meta = []

    # get remaining files for preprocessing
    pg_docs = dict()
    for filename in os.listdir(out_dir):
        pg_id = filename[3:-4]
        pg_docs[filename] = pg_id

    with open(path, 'r', encoding='utf-8') as file:  # get pre-cleaned gutenberg corpus
        cleaned_gutenberg = json.load(file)

        for sample in cleaned_gutenberg:
            pg_id = get_formatted_idx(sample['id'])  # PG-ID without '- extra'
            story = sample['text']
            title = sample['title']
            author = sample['author']

            if pg_id in pg_docs.values():
                preprocessor = pgp.PGPreprocessor(story, pg_id, title, author, out_dir)
                preprocessor.save_processed_text()

                entry = preprocessor.get_meta_data(f"JM-{pg_id}")
                pg_dataset_meta.append(entry)

    # save meta data accordingly
    save_metadata(pg_dataset_meta, final_gold_meta_path_pg)


if __name__ == "__main__":
    path = '../Dataset_Analysis/ProjectGutenberg/cleaned_gutenberg_8-16K.json'
    story_8_16_path = '../Dataset_Analysis/ProjectGutenberg/short_stories_8_16.json'

    final_gold_meta_path_pg = '../data/project_gutenberg/final_pg_gold_meta.json'
    out_dir = '../data/project_gutenberg/gold_dataset/'

    os.makedirs(out_dir) if not os.path.isdir(out_dir) else None

    # STEP 1 + 2:
    # RUN THIS TO GET THE RAW TEXT FILES OF ALL STORIES BETWEEN 8K AND 16K TOKENS IN out_dir
    # --> run_step_1_2(path, story_8_16_path, out_dir)
    # REMOVE OTHER NOT FITTING DOCUMENTS MANUALLY AFTERWARDS
    # either by adding keywords to this code or removing them by hand

    # STEP 3
    # AUTOMATIC PREPROCESSING OF REMAINING DOCUMENTS
    run_step_3(out_dir, path, final_gold_meta_path_pg)

    # STEP 4
    # manual adjustments by hand to introduce a streamlined format
