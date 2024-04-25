# LLaMA Scripts

## Preparation
```bash
cd llama
pip install poetry
poetry install
cd src/medical-evaluation
poetry run pip install -e .[multilingual]
poetry run python -m spacy download en_core_web_lg
```

## Automated Task Assessment
Before evaluation, please download [BART checkpoint](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) to `src/metrics/BARTScore/bart_score.pth`.

To generate predictions and corresponding gold labels which are saved in JSON format, please follow these instructions.

To use a model hosted on the HuggingFace Hub (for instance, llama2-7b-hf), change this command in `scripts/run_evaluation.sh`:

```bash
poetry run python src/eval.py \
    --model "hf-causal-vllm" \
    --model_args "use_accelerate=True,pretrained=[MODEL],use_fast=False" \
    --tasks "[TASK]"
    --write_out
```
The system will default to saving the file in the current folder.

### Tasks

We now support following tasks:

| Data                  | Shots                             | 
| --------------------- | -------------------------------- |
| MS2 | 0 |
| LitCovid | 0 |
| HoC | 0 |
| MedQA | 0 |
| PubmedQA | 0 |
| PubmedSum | 0 |
| CochranePLS | 0 |
| PLOS | 0 |
| MS21Shot | 1 |
| LitCovid1Shot | 1 |
| HoC1Shot | 1 |
| MedQA1Shot | 1 |
| PubmedQA1Shot | 1 |
| PubmedSum1Shot | 1 |
| CochranePLS1Shot | 1 |
| PLOS1Shot | 1 |

### Models

To replicate results, please use following models:

| Model | Finetuned |
| --------------------- | -------------------------------- |
|meta-llama/Llama-2-13b-chat-hf|False|
|chaoyi-wu/MedLLaMA_13B|False|
| clinicalnlplab/finetuned-Llama-2-13b-hf-CochranePLS | True    |
| clinicalnlplab/finetuned-Llama-2-13b-hf-HoC         | True    |
| clinicalnlplab/finetuned-Llama-2-13b-hf-LitCovid    | True    |
| clinicalnlplab/finetuned-Llama-2-13b-hf-MS2         | True    |
| clinicalnlplab/finetuned-Llama-2-13b-hf-MedQA       | True    |
| clinicalnlplab/finetuned-Llama-2-13b-hf-PLOS        | True    |
| clinicalnlplab/finetuned-Llama-2-13b-hf-PubmedQA    | True    |
| clinicalnlplab/finetuned-LLaMA-2-13b-hf-PubmedSumm  | True    |
| clinicalnlplab/finetuned-PMCLLaMA-13B-CochranePLS   | True    |
| clinicalnlplab/finetuned-PMCLLaMA-13B-HoC           | True    |
| clinicalnlplab/finetuned-PMCLLaMA-13B-MS2           | True    |
| clinicalnlplab/finetuned-PMCLLaMA-13B-MedQA         | True    |
| clinicalnlplab/finetuned-PMCLLaMA-13B-PubmedQA      | True    |
| clinicalnlplab/finetuned-PMCLLaMA-13b-LitCovid      | True    |
| clinicalnlplab/finetuned-PMCLLaMA-PLOS              | True    |
| clinicalnlplab/finetuned-PMCLLaMA-PubmedSumm        | True    |

