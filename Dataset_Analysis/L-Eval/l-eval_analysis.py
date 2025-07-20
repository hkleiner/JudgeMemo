import Dataset_Analysis.DatasetAnalyser as da
from datasets import load_dataset


if __name__ == "__main__":
    dataset = "sci_fi"
    # The corresponding NAMEs in the paper "SFiction"

    # disable_caching()  uncomment this if you cannot download codeU and sci_fi
    data = load_dataset('L4NLP/LEval', dataset, split='test')

    data_analyzer = da.DatasetAnalyser(
        min_tokens=8000,
        max_tokens=16000,
        clean=False,
        dataset=data
    )

    data_analyzer.run_analysis(
        idx_key='',
        content_key='input'
    )
