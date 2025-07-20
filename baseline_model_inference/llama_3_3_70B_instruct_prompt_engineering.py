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

# Model and inference configuration
MODEL = "Llama-3.3-70B-Instruct"
MODEL_NAME = "meta-llama/" + MODEL
MAX_GEN_LEN = 1024
TEMP = 0.0
N_GPUs = 2

# Load the LLM model with specified options
llm_model = LLM(
    model=MODEL_NAME, 
    tensor_parallel_size=N_GPUs, 
    enable_prefix_caching=True, 
    quantization="fp8",
    download_dir="/dss/mcmlscratch/00/di38riv"
)

sampling_params = SamplingParams(temperature=TEMP, max_tokens=MAX_GEN_LEN)
print("--> Model loaded successfully")

# Dataset and experimental settings
dataset = "project_gutenberg"  # test sub_sample
mode = "gold"  # gold manipulated
subset = "pg-1900"  # full sanity test
subset_num = "full"  # 1000 2000 3000 full
metric = "F"  # C F
category = "4"
mani = "43"  # 45 44 43 41 | 11 12 13 14
exp_number = 1
RANGE = "document"

# Define the prompt version and load the prompt template from file
prompt_versions = ["v6-2_fluency", "v6-2_coherence", "v6-3"]
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
    for i, story in enumerate(stories):
        full_prompt = USER_TEMPLATE_PROMPT.format(Story=story)

        # Determine output path for prompt files
        if mode == "gold":  # GOLD
            out_prompt = f"../experiments/full_prompts/{dataset}_{prompt_version}/{mode}_{subset}_{subset_num}/prompt_gold_{ids[i]}.txt"
        else:  # MANIPULATED
            if metric == "C":  # COHERENCE
                out_prompt = f"../experiments/full_prompts/{dataset}_{prompt_version}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/{RANGE}/prompt_exp_{exp_number}_{ids[i]}.txt"
            else:  # FLUENCY
                out_prompt = f"../experiments/full_prompts/{dataset}_{prompt_version}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/prompt_exp_{exp_number}_{ids[i]}.txt"

        # Create directories if not existing and save the prompt text
        if not os.path.isdir(os.path.dirname(out_prompt)):
            os.makedirs(os.path.dirname(out_prompt))
        with open(out_prompt, 'w', encoding='utf-8') as out_prompt:
            out_prompt.write(full_prompt)

        # Prepare conversation format for LLM input
        # Each conversation consists of a system message and a user message
        conversation = [
            {
                "role": "system",
                "content": """You are a human annotator that rates the quality of texts."""
            },
            {
                "role": "user",
                "content": full_prompt
            }
        ]
        prompts.append(conversation)
    print(f"--> Loaded prompts for experiment {subset_num}-{prompt_version}-{metric}/{mani}/{exp_number} with {len(prompts)} conversations")

    # Generate outputs
    outputs = llm_model.chat(messages=prompts,
                             sampling_params=sampling_params,
                             use_tqdm=True)

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