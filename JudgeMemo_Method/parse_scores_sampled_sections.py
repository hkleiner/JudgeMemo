import os
import pandas as pd
from JudgeMemo.JMParser import JMParser

import numpy as np
from scipy import stats


def parse_scores_from_directory(base_path: str, output_csv_path: str):
    parser = JMParser("ScoreParsing")
    data = {}

    all_section_names = set()

    for document_id in sorted(os.listdir(base_path)):
        doc_path = os.path.join(base_path, document_id)
        if not os.path.isdir(doc_path):
            continue

        col_name = f"{document_id[15:]}"
        data[col_name] = {}

        for filename in sorted(os.listdir(doc_path)):
            if filename.endswith(".txt"):
                section_name = os.path.splitext(filename)[0]  # e.g., section_3
                all_section_names.add(section_name)
                file_path = os.path.join(doc_path, filename)

                parsed = parser.parse(input_data=file_path, is_raw_text=False, scores_only=True)
                if parsed is not None and not parsed.empty:
                    row = parsed.iloc[:, 0]  # first column of parsed scores
                    flu = row.get("fluency", "")
                    coh = row.get("coherence", "")

                    data[col_name][(section_name, "fluency")] = flu
                    data[col_name][(section_name, "coherence")] = coh

    # Create rows indexed by (section, metric)
    all_rows = []
    all_section_names = sorted(all_section_names, key=lambda x: int(x.split("_")[-1]))  # natural order
    for section in all_section_names:
        for metric in ["fluency", "coherence"]:
            row = {
                "document_id": section,
                "Metric": metric
            }
            for col in data:
                row[col] = data[col].get((section, metric), "")
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved pivoted score table to: {output_csv_path}")


if __name__ == "__main__":
    prompt = "v6-3"
    model = "Llama-3.3-70B-Instruct"

    # SETTINGS
    REPORT_MODE = "report_only"
    INCLUDE_SEC_SUMMARIES = False
    SEC2TAG = True
    SCAN_RANGE = 2000
    SCAN_OVERLAP_RATIO = 0.0
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
            input_dir = f"../experiments_JM/{model}/{dataset}_{prompt}/{SETTINGS}/{mode}_{subset}_{subset_num}/sec_evaluations/"
            output_csv = f"scores_{mode}_{SETTINGS}.csv"
        else:
            if metric == "C":
                input_dir = f"../experiments_JM/{model}/{dataset}_{prompt}/{SETTINGS}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/{range_}/sec_evaluations/"
                output_csv = f"scores_{mode}_{SETTINGS}-{mani}.csv"
            else:
                input_dir = f"../experiments_JM/{model}/{dataset}_{prompt}/{SETTINGS}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/sec_evaluations/"
                output_csv = f"scores_{mode}_{SETTINGS}-{mani}.csv"

        combined_df = parse_scores_from_directory(input_dir, output_csv)
