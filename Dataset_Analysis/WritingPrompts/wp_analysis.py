import Dataset_Analysis.DatasetAnalyser as da
from datasets import load_dataset


if __name__ == "__main__":
    writing_prompts = load_dataset("euclaise/writingprompts", split="train")

    dataset_analyzer = da.DatasetAnalyser(
        min_tokens=8000,
        max_tokens=16000,
        dataset=writing_prompts
    )

    dataset_analyzer.run_analysis(
        idx_key='prompt',
        content_key='story'
    )
