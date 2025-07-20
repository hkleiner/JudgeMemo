import os

# Set environment variables for Hugging Face cache and home directories.
# Adjust these paths according to your system's storage configuration.
os.environ["HF_HUB_CACHE"] = "/path/to/your/storage"
os.environ["HF_HOME"] = "/path/to/your/storage"

from huggingface_hub import login

# Log in to Hugging Face Hub using a token stored in environment variables.
login(os.environ["HUGGINGFACE_TOKEN"])

import gc
import torch
import re

import json
from vllm import LLM, SamplingParams

from vllm.distributed.parallel_state import destroy_model_parallel


def is_output_complete(text: str) -> bool:
    """
    Check if the model's output is complete (finish reason = 'stop').

    Returns False if 'finish_reason: length' is found in the text,
    meaning the model output was cut off and needs to be rerun.
    Otherwise, returns True.
    """
    return not bool(re.search(r"^\s*finish_reason:\s*length\s*$", text, re.MULTILINE))


# Model and inference configuration
THINK = True  # True for thinking mode, False for non-thinking mode
MODEL = "Llama-3_3-Nemotron-Super-49B-v1" 
MODEL_NAME = "nvidia/" + MODEL
MODEL = MODEL if not THINK else MODEL + "_THINK"
thinking = "on" if THINK else "off"
MAX_GEN_LEN = 2048 if THINK else 1024
TEMP = 0.6 if THINK else 0.0
N_GPUs = 2
TOP_P = 0.95 if THINK else 1.0  # what to set top_p to for non-thinking mode?

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
    top_p=TOP_P
)
print("--> Model loaded successfully")

# Define the prompt version and load the prompt template from file
prompt_version = "v6-3"
prompt_path = f"./prompts/prompt_{prompt_version}.txt"

with open(prompt_path, 'r', encoding='utf-8') as prompt_file:
    USER_TEMPLATE_PROMPT = prompt_file.read()

# Dataset and experimental settings
dataset = "project_gutenberg"  # test sub_sample
subset = "pg-1900"  # full sanity test
subset_num = "2000"  # 2000

# Define experimental configurations (modes, metrics, categories, etc.)
configs = [
    {"mode": "gold",        "metric": "",  "category": "",  "mani": "",  "exp_number": 1, "range": ""},
    {"mode": "manipulated", "metric": "F", "category": "4", "mani": "43","exp_number": 1, "range": "document"},
    {"mode": "manipulated", "metric": "F", "category": "4", "mani": "42","exp_number": 1, "range": "document"},
    {"mode": "manipulated", "metric": "F", "category": "4", "mani": "41","exp_number": 1, "range": "document"},
    {"mode": "manipulated", "metric": "C", "category": "3", "mani": "32","exp_number": 1, "range": "document"},
    {"mode": "manipulated", "metric": "C", "category": "2", "mani": "24","exp_number": 1, "range": "paragraph"},
    {"mode": "manipulated", "metric": "C", "category": "1", "mani": "17","exp_number": 1, "range": "paragraph"},
]

# Iterate over each experimental setup
for config in configs:
    mode, metric, category, mani, exp_number, range_ = config
    print(f"--> Processing experiment {subset}-{subset_num}-{prompt_version}-{metric}-{mode}/{mani}/{exp_number}...")

    # Prepare prompts
    prompts = []
    stories = []
    ids = []

    # Determine the data file path depending on mode and metric
    if mode == "gold":  # GOLD
        path = f"../data/{dataset}/{mode}_{subset}_meta_{subset_num}.json"
    else:  # MANIPULATED
        if metric == "C":  # COHERENCE
            path = f"../data/{dataset}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/{range_}/exp_{exp_number}_meta.json"
        else:  # FLUENCY
            path = f"../data/{dataset}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/exp_{exp_number}_meta.json"

    # Load metadata JSON file describing documents to process
    with open(path, 'r', encoding='utf-8') as data:
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
                "content": f"detailed thinking {thinking}"
            },
            {
                "role": "user",
                "content": "You are a human annotator that rates the quality of texts.\n\n" + USER_TEMPLATE_PROMPT.format(Story=story)
            }
        ]
        prompts.append(conversation)
    print(f"--> Created all conversations for experiment {subset}-{subset_num}-{prompt_version}-{metric}-{mode}/{mani}/{exp_number}")

    # Generate model outputs with retry logic for incomplete outputs
    MAX_RETRIES = 3
    completed_responses = [None] * len(prompts)
    remaining_indices = list(range(len(prompts)))
    attempt = 0

    while remaining_indices and attempt <= MAX_RETRIES:
        print(f"--> Generation attempt {attempt + 1} for {len(remaining_indices)} texts...")

        # Select prompts still needing generation
        current_prompts = [prompts[i] for i in remaining_indices]
        # Generate model outputs
        outputs = llm_model.chat(messages=current_prompts, sampling_params=sampling_params, use_tqdm=True)

        next_remaining = []
        # Evaluate outputs for completeness
        for i, output in enumerate(outputs):
            idx = remaining_indices[i]
            response = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            # Append finish reason to the response for completeness check
            response_with_reason = response + f"\n\n----------------------------\nfinish_reason: {finish_reason}"
            # If output is complete, save it; else mark for retry
            if is_output_complete(response_with_reason):
                completed_responses[idx] = response_with_reason
            else:
                next_remaining.append(idx)

        remaining_indices = next_remaining
        attempt += 1

    # Save all responses — complete or final retry output
    for i in range(len(completed_responses)):
        if mode == "gold":
            out_path_dir = f"../experiments/{MODEL}/{dataset}_{prompt_version}/{mode}_{subset}_{subset_num}/"
        else:
            if metric == "C":
                out_path_dir = f"../experiments/{MODEL}/{dataset}_{prompt_version}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/{range_}/"
            else:
                out_path_dir = f"../experiments/{MODEL}/{dataset}_{prompt_version}/{mode}_{subset}_{subset_num}/{metric}/{category}/{mani}/"

        if not os.path.isdir(out_path_dir):
            os.makedirs(out_path_dir)

        output_path = os.path.join(out_path_dir, f"eval_exp_{exp_number}_{ids[i]}.txt")

        if completed_responses[i] is not None:
            # Write the successfully completed output to file
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(completed_responses[i])
        else:
            # If not completed, generate output one last time without tqdm and save with a warning
            last_retry_prompt = [prompts[i]]
            last_output = llm_model.chat(messages=last_retry_prompt, sampling_params=sampling_params, use_tqdm=False)[0]
            final_text = last_output.outputs[0].text
            finish_reason = last_output.outputs[0].finish_reason
            response_with_reason = final_text + f"\n\n----------------------------\nfinish_reason: {finish_reason}"

            if is_output_complete(response_with_reason):
                output_text = response_with_reason
            else:
                output_text = response_with_reason + "\nWARNING: INCOMPLETE OUTPUT after max retries"

            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(output_text)

print("--> Processed all experiments")
# Clean up and free GPU memory after processing all experiments
destroy_model_parallel()
del llm_model
gc.collect()
torch.cuda.empty_cache()
torch.distributed.destroy_process_group()
print("Successfully delete the llm pipeline and free the GPU memory!")
