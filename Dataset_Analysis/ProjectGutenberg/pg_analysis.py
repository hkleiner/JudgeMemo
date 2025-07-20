import Dataset_Analysis.DatasetAnalyser as da
from datasets import load_dataset
import json


if __name__ == "__main__":
    load = True

    if load:
        # Load dataset with parallel processing
        gutenberg = load_dataset("manu/project_gutenberg", split="en")

        dataset_analyzer = da.DatasetAnalyser(
            min_tokens=16000,
            max_tokens=32000,
            dataset=gutenberg,
            clean=True
        )
        with open("cleaned_gutenberg_16-32K.json", 'w', encoding='utf-8') as file:
            json.dump(list(dataset_analyzer.dataset), file)

        dataset_analyzer.run_analysis(path_min_max_stories="./short_stories_16_32.json")
    else:
        cleaned_gutenberg = load_dataset("json", data_files="./cleaned_gutenberg.json", split="train")

        dataset_analyzer = da.DatasetAnalyser(
            min_tokens=8000,
            max_tokens=16000,
            dataset=cleaned_gutenberg,
            clean=False
        )
        dataset_analyzer.run_analysis(path_min_max_stories="./short_stories_8_16_new.json")
