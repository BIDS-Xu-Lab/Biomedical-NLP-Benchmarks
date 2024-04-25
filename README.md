# Biomedical Natural Language Processing Benchmarks


## Benchmarks
The datasets are biomedical natural language processing (BioNLP) benchmarks commonly adopted for benchmarking BioNLP lanuage models. It consists of the following:
- **The original full dataset:** For each `dataset_name`, the original complete training (train), development (dev), and testing (test) datasets are located in the `benchmarks/{dataset_name}/datasets/full_set/` directory. These datasets have been prepared based on existing studies. The train and dev files were used to fine-tune models.
- **Prompts**: For each `dataset_name`, zero-shot and one-shot prompts are located in the `benchmarks/{dataset_name}/` directory. We selected one fixed example from the train file for one-shot learning.

## Fine-tuning for Llama models

Please adhere to the instructions in the `LLMindCraft` submodule folder, which provides both the preprocessing scripts and fine-tuning docker images.

We also provide the preprocessed datasets for fine-tuning:

| Dataset                                           |
|---------------------------------------------------|
| clinicalnlplab/CochranePLS_train                  |
| clinicalnlplab/HoC_train                          |
| clinicalnlplab/LitCovid_train                     |
| clinicalnlplab/MS2_train                          |
| clinicalnlplab/MedQA_train                        |
| clinicalnlplab/PLOS_train                         |
| clinicalnlplab/PubmedQA_train                     |
| clinicalnlplab/PubmedSumm_train                   |


## Running the prediction script for GPT models

To generate predictions for 6 generative tasks (**[QA]MedQA(5-option)**, **[QA]PubMedQA**, **[Summarization]PubMed**, **[Summarization]MS^2**, **[Simplification]Cochrane**, **[Simplification]PLOS**), please use the following command:

```bash
python generative_tasks/run_gpt.py \
 --dataset {medqa5 | pubmedqa | pubmed | ms2 | cochrane | plos} \
 --model {gpt-35-turbo-16k | gpt-4-32k } \
 --setting {zero_shot | one_shot}
```
Predictions and corresponding gold labels are saved in JSON format, for example, `ms2_gpt-4-32k_one_shot.json`. The JSON files include both the predicted outputs and the gold standard labels for all examples within this dataset.

To generate predictions for 6 extractive tasks (**[NER]BC5CDR-chemical**, **[NER]NCBI Disease**, **[RE]ChemProt**, **[RE]DDI2013**, **[MLC]HoC**,  **[MLC]LitCovid**), please use the following command:

```bash
python extractive_tasks/run_gpt.py
```
and
```bash
python extractive_tasks/run_convert_pred_2_json.py
```
to generate all predictions (6 extractive tasks for GPT-3.5 / 4, zero_shot / one_shot) all together. Predictions and corresponding gold labels are saved in JSON format, for example, `Hoc_gpt4_os.json`. The JSON files include both the predicted outputs and the gold standard labels for all examples within this dataset.

## Running the prediction script for Llama models

Please adhere to the instructions in the `llama` folder. Note that the evaluation script within this folder serves merely as a reference. For consistent results across all models — including Llama models and GPT models — we used `run_eval.py` for evaluations.

Predictions and corresponding gold labels are saved in JSON format, for example, `ms2_llama2_13b_chat_one_shot.json`. The JSON files include both the predicted outputs and the gold standard labels for all examples within this dataset.

## Evaluation

Before evaluation, please download [BART checkpoint](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) (for BART metrics evaluation).

To evaluate on various datasets or tasks, please use the following command:
```bash
python run_eval.py \
 --json_file {ms2_gpt-4-32k_one_shot.json | ms2_llama2_13b_chat_one_shot.json | ...} \
 --format_type {gpt | llama} \ 
 --task {NER | RE | MLC | QA | summarization | simplification}
```


## Results

|             | Main metrics |SOTA results before LLMs | GPT-3.5 zero-shot | GPT-4 zero-shot  | LLAMA2 13B zero-shot | GPT-3.5 one-shot | GPT-4 one-shot  | LLAMA2 13B one-shot | LLAMA2 13B fine-tuned | PMC LLAMA 13B fine-tuned |
|-------------|-----|---------------------|---------|---------|------------|---------|---------|------------|------------|---------------|
| [NER]BC5CDR-chemical     | Entity F1                | 0.9500  | 0.6274  | **0.7993***    | 0.3944  | 0.7133  | **0.8327***    | 0.6276     | **0.9149**        | 0.9063        |
| [NER]NCBI Disease        | Entity F1                | 0.9090  | 0.4060  | **0.5827***    | 0.2211  | 0.4817  | **0.5988***    | 0.3811     | **0.8682***       | 0.8353        |
| [RE]ChemProt             | Macro F1                 | 0.7344  | 0.1345  | **0.3250***    | 0.1392  | 0.1280  | **0.3391***    | 0.0718     | **0.4612***       | 0.3111        |
| [RE]DDI2013              | Macro F1                 | 0.7919  | 0.2004  | **0.2968***    | 0.1305  | 0.2126  | **0.3312***    | 0.1779     | **0.6218**        | 0.5700        |
| [MLC]HoC                 | Macro F1                 | 0.8882  | 0.6722  | **0.7109***     | 0.1285  | 0.6671  | **0.7093***     | 0.3072     | **0.6957***       | 0.4221        |
| [MLC]LitCovid            | Macro F1                 | 0.8921  | **0.5967**  | 0.5883     | 0.3825  | **0.6009**  | 0.5901     | 0.4808     | **0.5725***       | 0.4273        |
| [QA]MedQA(5-option)      | Accuracy                 | 0.4195  | 0.4988  | **0.7156***    | 0.2522  | 0.5161  | **0.7439***    | 0.2899     | **0.4462***       | 0.3975        |
| [QA]PubMedQA             | Accuracy                 | 0.7340  | **0.6560*** | 0.6280     | 0.5520  | 0.4600  | **0.7100***    | 0.2660     | **0.8040***       | 0.7680        |
| [Summarization]PubMed    | Rouge-L                  | 0.4316  | 0.2274  | **0.2419***    | 0.1190  | 0.2351  | **0.2427***    | 0.0989     | **0.1857***       | 0.1684        |
| [Summarization]MS^2      | Rouge-L                  | 0.2080  | 0.0889  | **0.1224***    | 0.0948  | 0.1132  | **0.1248***    | 0.0320     | **0.0934***       | 0.0059        |
| [Simplification]Cochrane | Rouge-L                  | 0.4476  | 0.2365  | **0.2375**     | 0.2081  | **0.2447*** | 0.2385     | 0.2207     | 0.2355        | **0.2370***       |
| [Simplification]PLOS     | Rouge-L                  | 0.4368  | **0.2323*** | 0.2253     | 0.2121  | **0.2449*** | 0.2386     | 0.1836     | **0.2583**        | 0.2577        |
|                          | Macro-average            | 0.6536  | 0.3814  | **0.4561**     | 0.2362  | 0.3848  | **0.4750**     | 0.2614     | **0.5131**        | 0.4422        |


## Original repository

https://github.com/qingyu-qc/gpt_bionlp_benchmark
