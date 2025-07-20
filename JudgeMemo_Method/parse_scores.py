import os
import pandas as pd
from JudgeMemo.JMParser import JMParser

import numpy as np
from scipy import stats


def parse_scores_from_directory(directory_path: str):
    """
    Parses score files from a directory, aggregates them, and computes statistical metrics.

    Each file is parsed using JMParser. Scores are combined into a
    single DataFrame, from which the mean, standard deviation, and 95% confidence intervals
    are computed across documents.

    Args:
        directory_path (str): Path to the directory containing `.txt` score files.

    Returns:
        pandas.DataFrame: Combined DataFrame with per-score statistics added:
            - 'mean': Mean score per row across files.
            - 'std': Standard deviation.
            - 'ci_lower' / 'ci_upper': 95% confidence interval range.
            Returns `None` if no valid files are found.
    """
    parser = JMParser("ScoreParsing")
    all_scores = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            df = parser.parse(input_data=file_path, is_raw_text=False, scores_only=True)
            all_scores.append(df)

    if all_scores:
        combined_df = pd.concat(all_scores, axis=1)

        # Calculate mean and std
        mean_col = combined_df.mean(axis=1)
        std_col = combined_df.std(axis=1)

        # Number of files (for CI)
        n = combined_df.shape[1]
        sem = std_col / np.sqrt(n)
        ci_range = stats.t.ppf(0.975, df=n - 1) * sem  # 95% CI using t-distribution

        # Add new columns
        combined_df["mean"] = mean_col
        combined_df["std"] = std_col
        combined_df["ci_lower"] = mean_col - ci_range
        combined_df["ci_upper"] = mean_col + ci_range

        return combined_df
    else:
        print(f"No valid files found in {directory_path}.")
        return None


if __name__ == "__main__":
    prompt = "v6-3"
    model = "Qwen3-32B_THINK"

    # SETTINGS
    REPORT_MODE = "report_only"
    INCLUDE_SEC_SUMMARIES = False
    SEC2TAG = True
    SCAN_RANGE = 2000
    SCAN_OVERLAP_RATIO = 0.1
    SETTINGS = f"sr-{SCAN_RANGE}_sor-{SCAN_OVERLAP_RATIO}_iss-{INCLUDE_SEC_SUMMARIES}_rm-{REPORT_MODE}_s2t-{SEC2TAG}"

    dataset = "JM-sub-sample"
    subset = "pg-1900"
    subset_num = "full"
    configs = [
        {"mode": "gold", "metric": "", "category": "", "mani": "", "exp_number": 1, "range": ""},
        {"mode": "manipulated", "metric": "F", "category": "4", "mani": "43", "exp_number": 1, "range": "document"},
        {"mode": "manipulated", "metric": "F", "category": "4", "mani": "41", "exp_number": 1, "range": "document"},
        {"mode": "manipulated", "metric": "F", "category": "4", "mani": "42", "exp_number": 1, "range": "document"},
        {"mode": "manipulated", "metric": "C", "category": "2", "mani": "24", "exp_number": 1, "range": "paragraph"},
        {"mode": "manipulated", "metric": "C", "category": "1", "mani": "17", "exp_number": 1, "range": "paragraph"},
        {"mode": "manipulated", "metric": "C", "category": "3", "mani": "32", "exp_number": 1, "range": "document"},
    ]

    all_dfs = []
    for cfg in configs:
        mode = cfg["mode"]
        metric = cfg["metric"]
        category = cfg["category"]
        mani = cfg["mani"]
        exp_number = cfg["exp_number"]
        range_ = cfg["range"]

        if mode == "gold":
            input_dir = f"../experiments_JM/{model}/{dataset}_{prompt}/{SETTINGS}/{mode}_{subset}_{subset_num}/"
            output_csv = f"scores_{mode}_{SETTINGS}.csv"
        else:
            if metric == "C":
                input_dir = f"../experiments_JM/{model}/{dataset}_{prompt}/{SETTINGS}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/{range_}/"
                output_csv = f"scores_{mode}_{SETTINGS}-{mani}.csv"
            else:
                input_dir = f"../experiments_JM/{model}/{dataset}_{prompt}/{SETTINGS}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/"
                output_csv = f"scores_{mode}_{SETTINGS}-{mani}.csv"

        combined_df = parse_scores_from_directory(input_dir, output_csv)

        if combined_df is not None:
            config_str = f"{mode}_{metric}_{category}_{mani}_{range_}_{prompt}"
            combined_df.insert(0, "config", config_str)
            all_dfs.append(combined_df)

    if all_dfs:
        final_df = pd.concat(all_dfs, axis=0)
        final_df.to_csv(f"all_combined_scores_{model}_{SETTINGS}.csv", index=False)
        print("Saved final combined DataFrame.")
    else:
        print("No data to save.")
