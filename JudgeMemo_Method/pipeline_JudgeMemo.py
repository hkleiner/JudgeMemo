import os
os.environ["HF_HUB_CACHE"] = "/dss/mcmlscratch/00/di38riv"
os.environ["HF_HOME"] = "/dss/mcmlscratch/00/di38riv"

from huggingface_hub import login
login(os.environ["HUGGINGFACE_TOKEN"])

import gc
import torch

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel


# --MODEL PARAMS-
THINK = True
# MODEL = "Llama-3_3-Nemotron-Super-49B-v1" 
# MODEL_NAME = "nvidia/" + MODEL
MODEL = "Qwen3-32B"
MODEL_NAME = "Qwen/" + MODEL
# MODEL = "Llama-3.3-70B-Instruct"
# MODEL_NAME = "meta-llama/" + MODEL
MODEL = MODEL if not THINK else MODEL + "_THINK"
thinking = "on" if THINK else "off"
MAX_GEN_LEN = 2048 if THINK else 1024
TEMP = 0.6 if THINK else 0.0  # Llama-3_3-Nemotron-Super-49B-v1
# TEMP = 0.6 if THINK else 0.7 # Qwen3-32B
# TEMP = 0.0 # Llama-3.3-70B-Instruct
TOP_K = 20 # Qwen3-32B
N_GPUs = 2
TOP_P = 0.95 if THINK else 1.0

llm_model = LLM(
    MODEL_NAME,
    tensor_parallel_size=N_GPUs,
    enable_prefix_caching=True,
    quantization="fp8",
    download_dir="/dss/mcmlscratch/00/di38riv",
    trust_remote_code=True
)
sampling_params = SamplingParams(
    temperature=TEMP,
    max_tokens=MAX_GEN_LEN, 
    top_p=TOP_P,
    top_k=TOP_K  # Qwen3-32B
)
print("--> Model loaded successfully")


from JudgeMemo.JMProcessor import JMProcessor
from JudgeMemo import JMUtils


# SETTINGS
REPORT_MODE = "report_only"
INCLUDE_SEC_SUMMARIES = False
SEC2TAG = True
SCAN_RANGE = 2000  # also try 3000 and 1000
SCAN_OVERLAP_RATIO = 0.1  # hard - 0.0; stride - > 0.1
SETTINGS = f"sr-{SCAN_RANGE}_sor-{SCAN_OVERLAP_RATIO}_iss-{INCLUDE_SEC_SUMMARIES}_rm-{REPORT_MODE}_s2t-{SEC2TAG}"

# --PROMPT TEMPLATE--
eval_sys = """You are a human annotator that rates the quality of texts."""  # Llama-3.3-70B-Instruct and Qwen3-32B
# eval_sys = f"detailed thinking {thinking}"  # Llama-3_3-Nemotron-Super-49B-v1

# --DATASET PARAMS--
dataset = "JM-sub-sample"  # test sub_sample
subset = "pg-1900"  # full sanity test
subset_num = "full"  # 1000 2000 3000

configs = [
    {"mode": "gold",        "metric": "",  "category": "",  "mani": "",  "exp_number": 1, "range": ""},
    {"mode": "manipulated", "metric": "F", "category": "4", "mani": "43","exp_number": 1, "range": "document"},
    {"mode": "manipulated", "metric": "F", "category": "4", "mani": "42","exp_number": 1, "range": "document"},
    {"mode": "manipulated", "metric": "F", "category": "4", "mani": "41","exp_number": 1, "range": "document"},
    {"mode": "manipulated", "metric": "C", "category": "3", "mani": "32","exp_number": 1, "range": "document"},
    {"mode": "manipulated", "metric": "C", "category": "2", "mani": "24","exp_number": 1, "range": "paragraph"},
    {"mode": "manipulated", "metric": "C", "category": "1", "mani": "17","exp_number": 1, "range": "paragraph"},
]

# --PROCESSING--
for cfg in configs:
    doc_mode    = cfg["mode"]
    metric      = cfg["metric"]
    category    = cfg["category"]
    mani        = cfg["mani"]
    exp_number  = cfg["exp_number"]
    range_      = cfg["range"]

    if doc_mode == "gold":  # GOLD
        path = f"../data/{dataset}/{doc_mode}_{subset}_meta_{subset_num}.json"
    else:  # MANIPULATED
        if metric == "C":  # COHERENCE
            path = f"../data/{dataset}/{doc_mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/{range_}/exp_{exp_number}_meta.json"
        else:  # FLUENCY
            path = f"../data/{dataset}/{doc_mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/exp_{exp_number}_meta.json"

    data_meta = JMUtils.load_meta_data(path)

    ids = []
    stories = []
    for elem in data_meta:
        if doc_mode == "gold":  # GOLD
            idx = elem['id']
            text_path = elem["gold_text"]
        else:  # MANIPULATED
            idx = elem['gold_id']
            text_path = elem["manipulated_text"]

        text = open(f"..{text_path}", 'r', encoding='utf-8').read()
        story = stories.append(text)
        ids.append(idx)

    for idx, story in enumerate(stories):  # all prompts to be processed
        if doc_mode == "gold":  # GOLD
            out_path_dir = f"../experiments_JM/{MODEL}/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/"
            sec_eval_path = f"../experiments_JM/{MODEL}/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/sec_evaluations/"
            memory_path = f"../experiments_JM/{MODEL}/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/memory/memory_file-{ids[idx]}.json"
            report_path = f"../experiments_JM/{MODEL}/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/reports/report_file-{ids[idx]}.txt"
            out_prompt = f"../experiments_JM/full_prompts/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/prompt_gold_{ids[idx]}.txt"
        else:  # MANIPULATED
            if metric == "C":  # COHERENCE
                out_path_dir = f"../experiments_JM/{MODEL}/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/{range_}/"
                sec_eval_path = f"../experiments_JM/{MODEL}/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/{range_}/sec_evaluations/"
                memory_path = f"../experiments_JM/{MODEL}/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/{range_}/memory/memory_file-{ids[idx]}.json"
                report_path = f"../experiments_JM/{MODEL}/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/{range_}/reports/report_file-{ids[idx]}.txt"
                out_prompt = f"../experiments_JM/full_prompts/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/{range_}/prompt_{ids[idx]}.txt"
            else:  # FLUENCY
                out_path_dir = f"../experiments_JM/{MODEL}/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/"
                sec_eval_path = f"../experiments_JM/{MODEL}/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/sec_evaluations/"
                memory_path = f"../experiments_JM/{MODEL}/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/memory/memory_file-{ids[idx]}.json"
                report_path = f"../experiments_JM/{MODEL}/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/reports/report_file-{ids[idx]}.txt"
                out_prompt = f"../experiments_JM/full_prompts/{dataset}_v6-3/{SETTINGS}/{doc_mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/prompt_{ids[idx]}.txt"

        if not os.path.isdir(out_path_dir):  # create output directory
            os.makedirs(out_path_dir)

        processor = JMProcessor(
            model=llm_model,
            model_name=MODEL_NAME,
            sampling_params=sampling_params,
            text=story,
            doc_id=ids[idx],
            THINK=THINK
        )

        final_evaluation = processor.process(
            include_sec_summaries=INCLUDE_SEC_SUMMARIES,
            report_mode=REPORT_MODE,
            sec2tag=SEC2TAG,
            scan_range=SCAN_RANGE,
            scan_overlap_ratio=SCAN_OVERLAP_RATIO,
            memory_path=memory_path,
            report_path=report_path,
            sec_eval_path=sec_eval_path,
            eval_sys=eval_sys,
            save_prompt_path=""  # no saving here else out_prompt
        )

        with open(out_path_dir + f"eval_exp_{exp_number}_{ids[idx]}.txt", 'w', encoding='utf-8') as file:
            file.write(final_evaluation)

del llm_model
torch.cuda.empty_cache()
gc.collect()
# torch.distributed.destroy_process_group()
destroy_model_parallel()
print("Successfully delete the llm pipeline and free the GPU memory!")
