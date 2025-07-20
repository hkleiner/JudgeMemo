import Dataset_Analysis.DatasetAnalyser as da


if __name__ == "__main__":
    path = "../../data/project_gutenberg/"

    dataset_analyzer = da.DatasetAnalyser(
        min_tokens=8000,
        max_tokens=16000,
        corpus_path=path
    )

    dataset_analyzer.run_analysis(path_min_max_stories='short_stories_8_16_gold_cleaned.json')
