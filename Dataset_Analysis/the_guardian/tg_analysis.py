import json

import Dataset_Analysis.DatasetAnalyser as da
from datasets import load_dataset


if __name__ == "__main__":
    guardian = load_dataset("Stefan171/TheGuardian-Articles", split="train")

    dataset_analyzer = da.DatasetAnalyser(
        min_tokens=8000,
        max_tokens=16000,
        dataset=guardian
    )

    with open("cleaned_guardian_8-16K.json", 'w', encoding='utf-8') as file:
        json.dump(list(dataset_analyzer.dataset), file)

    dataset_analyzer.run_analysis(
        idx_key="Article Title",
        content_key="Article Contents",
        path_min_max_stories="./short_stories_8_16_tg.json"
    )
