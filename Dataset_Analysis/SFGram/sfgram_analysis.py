import Dataset_Analysis.DatasetAnalyser as da


if __name__ == "__main__":
    book_contents = "./SFGram-dataset/book-contents/"

    dataset_analyzer = da.DatasetAnalyser(
        min_tokens=8000,
        max_tokens=16000,
        corpus_path=book_contents
    )

    dataset_analyzer.run_analysis(path_min_max_stories="./short_stories_8_16.json")
