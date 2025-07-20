import json

import Dataset_Analysis.DatasetAnalyser as da
from datasets import load_dataset


if __name__ == "__main__":
    time_magazine = load_dataset("SinclairSchneider/time_magazine", split="train")

    dataset_analyzer = da.DatasetAnalyser(
        min_tokens=8000,
        max_tokens=16000,
        dataset=time_magazine
    )

    with open("cleaned_timemagazine_8-16K.json", 'w', encoding='utf-8') as file:
        json.dump(list(dataset_analyzer.dataset), file)

    dataset_analyzer.run_analysis(
        idx_key='title',
        content_key='content',
        path_min_max_stories="./short_stories_8_16_tm.json"
    )
