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
    --model_args "use_accelerate=True,pretrained=meta-llama/Llama-2-7b-chat-hf,use_fast=False" \
    --tasks "MS2"
```
The system will default to saving the file in the current folder.
