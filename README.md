## JudgeMemo: A Memory-Based Method for Improving Long-Context LLM-as-a-Judge Assessments

### Abstract
As Large Language Models (LLMs) are increasingly adopted as automated evaluators (so-called LLMs-as-a-Judge) their ability to evaluate long-context inputs remains underexplored and underperforming. While technical advances have extended LLM context windows to over 100,000
tokens, empirical studies reveal significant performance degradation when reasoning across long
sequences. This thesis addresses this gap by introducing JUDGEMEMO, a structured memory-augmented evaluation framework inspired by human note-taking, designed to enhance LLM assessment reliability on long-form documents. We also present COFLUEVAL-LC, a novel diagnostic dataset built from Project Gutenberg texts, systematically manipulated to degrade fluency and
coherence across varied structural and granular levels. Through extensive experiments, we show
that traditional LLM-as-a-Judge setups struggle with long-context consistency and exhibit reduced
sensitivity to subtle quality manipulations. In contrast, JUDGEMEMO significantly improves evaluation performance and robustness by segmenting documents into smaller sub-units, generating
intermediate evaluations (memory), and feeding this structured memory back into the evaluation
loop. Our results demonstrate that memory-augmented setups can mitigate the contextual limitations of LLMs, offering a more human-aligned, scalable path toward reliable automatic evaluation
of long-form content.

This project is the result of a master's thesis conducted at LMU Munich (Ludwig-Maximilians-Universität München) within the interdisciplinary program of Computational Linguistics and Computer Science. The work was carried out at the Faculty of Language and Literature Studies as well as the Faculty of Mathematics, Informatics, and Statistics.

### Repository Contents
- `Dataset_Analysis/` — code for dataset analysis during data collection processes
- `Dataset_Creation/` — code for Project Gutenberg data preprocessing and dataset creation (gold and manipulated)
  - `Dataset_Creation/DatasetCreator/` — preprocessing package
  - `Dataset_Creation/templates/` — .json-templates for dataset creation
  - `Dataset_Creation/TextManipulation/` — data manipulation package
  - `Dataset_Creation/apply_manipulations_to_data.py` — script to manipulate gold documents
- `data/` — datasets and processed data files
- `baseline_model_inference/` — vanilla evaluation pipeline and prompt engineering scripts
  - `baseline_model_inference/prompts/` — all prompts needed to run our vanilla pipeline
- `experiments/` — experimental results for vanilla evaluation set up and prompt engineering
- `JudgeMemo_Method/` — code for our framework `JudgeMemo`
  - `JudgeMemo_Method/JudgeMemo/` — pipeline package
  - `JudgeMemo_Method/prompts/` — all prompts needed to run our pipeline (with context-awareness)
  - `JudgeMemo_Method/prompt_no-context/` — all prompts needed to run our pipeline (without context-awareness)
  - `JudgeMemo_Method/pipeline_JudgeMemo.py` — script to run our framework
- `experiments_JM/` — experimental results (outputs) for our method (`JudgeMemo`) and ablation studies
- `environment.yml` — Conda environment definition file  
- `requirements.txt` — Python dependencies for pip  
- `README.md` — This file, with instructions and overview  

### Requirements
- OS: Windows/Linux/macOS
- (recommended) GPU with CUDA support for faster inference
- for manipulating documents: `Python 3.11.5`
- for inference running: `Python 3.11.11`

### Setup Instructions
#### Clone the Repository
```bash
git clone https://github.com/hkleiner/JudgeMemo.git
cd JudgeMemo
```

#### Using Conda
```bash
conda create -n your_env_name python=3.11
conda activate your_env_name
pip install -r requirements.txt
```
#### Using pip
```bash
python -m venv your_env_name
myenv\Scripts\activate # on Windows
source myenv/bin/activate # on macOS/Linux
pip install -r requirements.txt
```

#### Vanilla Evaluation Inference
```bash
cd baseline_model_inference
# select a model script you want to run the pipeline with, e.g.
python3 llama_3_3_70B_instruct.py
# you may set the dataset/data subset and document length to evaluate on in-file
# depending on the model, you may set the reasoning mode in-file
```

#### JudgeMemo Inference
```bash
cd JudgeMemo_Method
# run inference with Llama-3.3-70B-Instruct
python3 pipeline_JudgeMemo.py
# you may set the dataset/data subset and document length to evaluate on in-file
```

### Contact
If you have any questions or want to get in touch, feel free to reach me at:

**University Contact (LMU Munich):** [H.Kleiner@campus.lmu.de](mailto:H.Kleiner@campus.lmu.de)

**Contact**: [hermine.kleiner@yahoo.de](mailto:hermine.kleiner@yahoo.de)

**LinkedIn:** [linkedin.com/in/hermine-kleiner-09479b279](https://www.linkedin.com/in/hermine-kleiner-09479b279/)  

### License
This project is licensed under the .... See the [LICENSE](./LICENSE) file for details.