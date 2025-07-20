import os
import json
from DatasetCreator.utils import save_metadata


def run_step_1_2(path, story_8_16_path, out_dir, final_gold_meta_path_tm):
    tm_dataset_meta = []
    with (open(path, 'r', encoding='utf-8') as file,
          open(story_8_16_path, 'r', encoding='utf-8') as f_story):
        stories = json.load(f_story)
        cleaned_tm = json.load(file)

        counter = 0
        for sample in cleaned_tm:
            for (title, _) in stories:
                tm_title = sample['title']
                story = sample['content']

                story = title.upper() + "\n\n\n" + story

                if title == tm_title:
                    file_path = f"{out_dir}TM-{counter}.txt"
                    with open(file_path, 'w', encoding='utf-8') as out:
                        out.write(story)
                    entry = {
                        'id': f'JM-{counter}',
                        'gold_text': file_path,
                        'gold_source': {
                            'dataset': "SinclairSchneider/time_magazine",
                            'huggingface': {
                                'url': 'https://huggingface.co/datasets/SinclairSchneider/time_magazine'
                            }
                        }
                    }
                    tm_dataset_meta.append(entry)
            counter += 1

    # save meta data accordingly
    save_metadata(tm_dataset_meta, final_gold_meta_path_tm)


if __name__ == "__main__":
    path = "../Dataset_Analysis/time_magazine/cleaned_timemagazine_8-16K.json"
    story_8_16_path = "../Dataset_Analysis/time_magazine/short_stories_8_16_tm.json"

    final_gold_meta_path_tm = "../data/time_magazine/gold_full_meta.json"
    out_dir = "../data/time_magazine/gold_full/"

    os.makedirs(out_dir) if not os.path.isdir(out_dir) else None

    # STEP 1 + 2:
    # RUN THIS TO GET THE RAW TEXT FILES OF ALL STORIES BETWEEN 8K AND 16K TOKENS IN out_dir
    # -->
    run_step_1_2(path, story_8_16_path, out_dir, final_gold_meta_path_tm)
