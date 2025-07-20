from datasets import load_dataset, Features, Value, Sequence
import Dataset_Analysis.DatasetAnalyser as da


if __name__ == "__main__":
    # Define the features schema
    ft = Features({
        "id": Value("int64"),
        "context": Value("string"),
        "input": Value("string"),
        "answer": Sequence(Value("string")),
        "options": Sequence(Value("string"))
    })

    # Load the dataset with the specified features
    dataset_en_sum = load_dataset("xinrongzhang2022/InfiniteBench", features=ft, split="longbook_sum_eng")
    dataset_en_qa = load_dataset("xinrongzhang2022/InfiniteBench", features=ft, split="longbook_qa_eng")
    dataset_en_mc = load_dataset("xinrongzhang2022/InfiniteBench", features=ft, split="longbook_choice_eng")

    print("---EN.SUM---")
    data_analyzer_ensum = da.DatasetAnalyser(
        min_tokens=8000,
        max_tokens=16000,
        dataset=dataset_en_sum
    )

    data_analyzer_ensum.run_analysis(
        content_key='context'
    )

    print("---EN.QA---")
    data_analyzer_qa = da.DatasetAnalyser(
        min_tokens=8000,
        max_tokens=16000,
        dataset=dataset_en_qa
    )

    data_analyzer_qa.run_analysis(
        content_key='context'
    )

    print("---EN.MC---")
    data_analyzer_enmc = da.DatasetAnalyser(
        min_tokens=8000,
        max_tokens=16000,
        dataset=dataset_en_mc
    )

    data_analyzer_enmc.run_analysis(
        content_key='context'
    )