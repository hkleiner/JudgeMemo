import os
os.environ["HF_HUB_CACHE"] = "/dss/mcmlscratch/00/di38riv"
os.environ["HF_HOME"] = "/dss/mcmlscratch/00/di38riv"

from huggingface_hub import login
login(os.environ["HUGGINGFACE_TOKEN"])

import gc
import torch

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel


MODEL = "Llama-3.3-70B-Instruct"
MODEL_NAME = "meta-llama/" + MODEL
MAX_GEN_LEN = 1024
TEMP = 0.0
N_GPUs = 2

print("Starting to load the model...")
# Load the model through vLLM
model = LLM(MODEL_NAME, tensor_parallel_size=N_GPUs, enable_prefix_caching=True, quantization="fp8",
                download_dir="/dss/mcmlscratch/00/di38riv")
print("--> Model loaded successfully")
sampling_params = SamplingParams(temperature=TEMP, max_tokens=MAX_GEN_LEN)
print("--> SamplingParams loaded successfully")


from JudgeMemo import JMUtils
from JudgeMemo.JMJudge import JMJudge
from JudgeMemo.JMParser import JMParser
from JudgeMemo.JMScanner import JMScanner
from JudgeMemo.JMMemoryCreator import JMMemoryCreator


text = JMUtils.load_prompt_template("../data/project_gutenberg/gold_dataset/PG-642.txt")
scan_technique = "stride"
doc_id = "PG-642"

scanner = JMScanner(
    tokenizer=MODEL_NAME,
    text=text,
    doc_id=doc_id
)

doc_sections = scanner.scan(
    scan_range=2000,
    overlap_ratio=0.1,
    scan_mode=scan_technique
)

judge = JMJudge(
    model=model,
    sampling_params=sampling_params,
    doc_id=doc_id
)

doc_sec_evaluations = judge.evaluate_sections(
    doc_sections=doc_sections,
    include_sec_summaries=False,
    sec_evaluation_template=JMUtils.load_prompt_template(f"./prompts/prompt_v6-3_JudgeMemo_{scan_technique}.txt"),
    eval_sys='''You are a human annotator that rates the quality of texts.''',
    scan_technique=scan_technique
)

creator = JMMemoryCreator(doc_id=doc_id)
creator.create_memory(
    scanner_results=doc_sec_evaluations, 
    memory_file=f"mem-{doc_id}.json",
    is_raw_text=True
)

# For section-wise detailed report
print(creator.create_report(sec2tag=True))

destroy_model_parallel()
del model
gc.collect()
torch.cuda.empty_cache()
torch.distributed.destroy_process_group()
print("Successfully delete the llm pipeline and free the GPU memory!")
