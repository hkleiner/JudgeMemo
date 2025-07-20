import os

# Set environment variables for Hugging Face cache and home directories.
# Adjust these paths according to your system's storage configuration.
os.environ["HF_HUB_CACHE"] = "/dss/mcmlscratch/00/di38riv"
os.environ["HF_HOME"] = "/dss/mcmlscratch/00/di38riv"

from huggingface_hub import login

# Log in to Hugging Face Hub using a token stored in environment variables.
login(os.environ["HUGGINGFACE_TOKEN"])

import gc
import torch

import json
from vllm import LLM, SamplingParams

from vllm.distributed.parallel_state import destroy_model_parallel

"""
For thinking mode (enable_thinking=True), use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0. DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions.
For non-thinking mode (enable_thinking=False), we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
"""
# Model and inference configuration
THINK = False
MODEL = "Qwen3-32B"
MODEL_NAME = "Qwen/" + MODEL
MODEL = MODEL if not THINK else MODEL + "_THINK"
MAX_GEN_LEN = 2048 if THINK else 1024  # 512
TEMP = 0.6 if THINK else 0.7 # 0.0  # greedy decoding
TOP_P = 0.95 if THINK else 0.8
TOP_K = 20
N_GPUs = 2

# Load the LLM model with specified options
llm_model = LLM(
    model=MODEL_NAME, 
    tensor_parallel_size=N_GPUs, 
    trust_remote_code=True,
    download_dir="/dss/mcmlscratch/00/di38riv"
)

sampling_params = SamplingParams(
    temperature=TEMP, 
    max_tokens=MAX_GEN_LEN,
    top_p=TOP_P,
    top_k=TOP_K
)
print("--> Model loaded successfully")

# Dataset and experimental settings
dataset = "test"  # test sub_sample
mode = "manipulated"  # gold manipulated
subset = "test"  # full sanity test
subset_num = "full"  # 1000 2000 3000 full
metric = "F"  # C F
category = "4"
mani = "43"  # 45 44 43 41 | 11 12 13 14
exp_number = 1
RANGE = "document"  

# Define the prompt version and load the prompt template from file
prompt_versions = ["v6-1", "v6-2_fluency", "v6-2_coherence", "v6-3", "v6-4_fluency", "v6-4_coherence"]
# Iterate over each prompt
for prompt_version in prompt_versions:
    prompt = f"./prompts/prompt_{prompt_version}.txt"

    # Prepare prompts
    prompts = []
    stories = []
    ids = []

    # Determine the data file path depending on mode and metric
    if mode == "gold":  # GOLD
        path = f"../data/{dataset}/{mode}_{subset}_meta_{subset_num}.json"
    else:  # MANIPULATED
        if metric == "C":  # COHERENCE
            path = f"../data/{dataset}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/{RANGE}/exp_{exp_number}_meta.json"
        else:  # FLUENCY
            path = f"../data/{dataset}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/exp_{exp_number}_meta.json"

    # Load metadata JSON file describing documents to process
    with (open(prompt, 'r', encoding='utf-8') as prompt_file,
          open(path, 'r', encoding='utf-8') as data):
        USER_TEMPLATE_PROMPT = prompt_file.read()
        data_meta = json.load(data)

    # Load each text file referenced in the metadata
    for elem in data_meta:
        if mode == "gold":  # GOLD
            idx = elem['id']
            text_path = elem["gold_text"]
        else:  # MANIPULATED
            idx = elem['gold_id']
            text_path = elem["manipulated_text"]

        text = open(f"..{text_path}", 'r', encoding='utf-8').read()
        story = stories.append(text)
        ids.append(idx)

    # Create prompt conversations based on the loaded texts and template
    for story in stories:
        # Prepare conversation format for LLM input
        conversation = [
            {
                "role": "system",
                "content": """You are a human annotator that rates the quality of texts."""
            },
            {
                "role": "user",
                "content": USER_TEMPLATE_PROMPT.format(Story=story) + (" \\think" if THINK else "\\no_think")
            }
        ]
        prompts.append(conversation)
    print(f"--> Loaded prompts for experiment {subset_num}-{prompt_version}-{metric}/{mani}/{exp_number} with {len(prompts)} conversations")

    # Generate outputs
    outputs = llm_model.chat(
        messages=prompts,
        sampling_params=sampling_params,
        use_tqdm=True
    )

    # Save all responses to disk
    responses = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason  # why generation stopped
        if mode == "gold":  # GOLD
            out_path_dir = f"../experiments/{MODEL}/{dataset}_{prompt_version}/{mode}_{subset}_{subset_num}/"
        else:  # MANIPULATED
            if metric == "C":  # COHERENCE
                out_path_dir = f"../experiments/{MODEL}/{dataset}_{prompt_version}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/{RANGE}/"
            else:  # FLUENCY
                out_path_dir = f"../experiments/{MODEL}/{dataset}_{prompt_version}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/"
        # Create output directories if they don't exist
        if not os.path.isdir(out_path_dir):
            os.makedirs(out_path_dir)
        with open(out_path_dir + f"eval_exp_{exp_number}_{ids[i]}.txt", 'w', encoding='utf-8') as file:
            file.write(response + f"\n\n----------------------------\nfinish_reason: {finish_reason}")

    print(f"--> Saved all results for experiment {subset_num}-{prompt_version}-{metric}/{mani}/{exp_number}")

print("--> Processed all experiments")
# Clean up and free GPU memory after processing all experiments
destroy_model_parallel()
del llm_model
gc.collect()
torch.cuda.empty_cache()
torch.distributed.destroy_process_group()
print("Successfully delete the llm pipeline and free the GPU memory!")
