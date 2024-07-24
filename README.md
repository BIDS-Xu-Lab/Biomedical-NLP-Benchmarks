# Large language models for biomedical natural language processing: benchmarks, baselines, and recommendations

This is the github repository for ["A systematic evaluation of large language models for biomedical natural language processing: benchmarks, baselines, and recommendations"](https://arxiv.org/pdf/2305.16326). The related data and codes are publicly available, described below.


## 1. Benchmarks and models
This study consists of 12 benchmarks from six biomedical natural language processing applications: named entity recognition, relation extraction, multi-label document classification, question answering, text summarization, and text simplification.

The benchmarks are under [benchmarks folder](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/tree/main/benchmarks), explained below. 

### 1.1 Original datasets
Each has a **full_set** folder consisting the original
training (train), development (dev), and testing (test) datasets are located in the `benchmarks/{dataset_name}/datasets/full_set/` directory from the existing studies. 

### 1.2 Prompts for zero- and few-shot
We also made the prompts used in the study publicly available. For each `dataset_name`, zero- and few-shot prompts are also provided in the `benchmarks/{dataset_name}/` directory. For instance, [one-shot for pubmedqa](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/edit/main/benchmarks/%5BQA%5DPubMedQA/prompt_oneshot.txt) has the following information:

``` 
TASK: Your task is to answer biomedical questions using the given abstract. Only output yes, no, or maybe as answer. 
INPUT: The input is a question followed by an abstract.
OUTPUT: Answer each question by providing one of the following options: yes, no, maybe.
Example
INPUT: Does hippocampal atrophy on MRI predict cognitive decline? ["To investigate whether the presence of hippocampal atrophy (HCA) on MRI in Alzheimer's disease (AD) leads to a more rapid decline in cognitive function. To investigate whether cognitively unimpaired controls and depressed subjects with HCA are at higher risk than those without HCA of developing dementia.", 'A prospective follow-up of subjects from a previously reported MRI study.', 'Melbourne, Australia.', 'Five controls with HCA and five age-matched controls without HCA, seven depressed subjects with HCA and seven without HCA, and 12 subjects with clinically diagnosed probable AD with HCA and 12 without HCA were studied. They were followed up at approximately 2 years with repeat cognitive testing, blind to initial diagnosis and MRI result.', 'HCA was rated by two radiologists blind to cognitive test score results. Cognitive assessment was by the Cambridge Cognitive Examination (CAMCOG).', 'No significant differences in rate of cognitive decline, mortality or progression to dementia were found between subjects with or without HCA.']
OUTPUT: no
Input: {Input}
Output:
```
The `example input and output` are from an instance from the training set. `{Input}` is an instance from the testing set for inference.

### 1.3 Preprocessed dataset for instruction fine-tuning
We also provide the preprocessed datasets for instruction fine-tuning via [here](https://huggingface.co/collections/clinicalnlplab/instruction-datasets-for-benchmark-66a1234b13bb4260ed8f278a).

| Dataset|Train/Dev|Test|
|-------------|-----|-----|
| [MLC]HoC                 | [Train/Dev](https://huggingface.co/datasets/clinicalnlplab/HoC_train)|[Test](https://huggingface.co/datasets/clinicalnlplab/HoC_test)|
| [MLC]LitCovid            | [Train/Dev](https://huggingface.co/datasets/clinicalnlplab/LitCovid_train)|[Test](https://huggingface.co/datasets/clinicalnlplab/LitCovid_test)|
| [QA]MedQA(5-option)      | [Train/Dev](https://huggingface.co/datasets/clinicalnlplab/MedQA_train)|[Test](https://huggingface.co/datasets/clinicalnlplab/medQA_test)|
| [QA]PubMedQA             | [Train/Dev](https://huggingface.co/datasets/clinicalnlplab/PubmedQA_train)|[Test](https://huggingface.co/datasets/clinicalnlplab/pubmedqa_test)|
| [Summarization]PubMed    | [Train/Dev](https://huggingface.co/datasets/clinicalnlplab/PubmedSumm_train)||[Test]|
| [Summarization]MS^2      | [Train/Dev](https://huggingface.co/datasets/clinicalnlplab/MS2_train)|[Test](https://huggingface.co/datasets/clinicalnlplab/MS2_test)|
| [Simplification]Cochrane | [Train/Dev](https://huggingface.co/datasets/clinicalnlplab/CochranePLS_train)|[Test](https://huggingface.co/datasets/clinicalnlplab/CochranePLS_test)|
| [Simplification]PLOS     | [Train/Dev](https://huggingface.co/datasets/clinicalnlplab/PLOS_train)|[Test](https://huggingface.co/datasets/clinicalnlplab/PLOS_test)|


## Instruction fine-tuned models
We also made the instruction fine-tuned models in the study publicly available via [here] (https://huggingface.co/collections/clinicalnlplab/fine-tuned-models-for-benchmark-662948bb459e07dc7cef959a).

| Dataset|LLAMA|PMC-LLAMA|
|-------------|-----|-----|
| [NER]BC5CDR-chemical     | [LLAMA 2 13B](https://huggingface.co/clinicalnlplab/finetuned-Llama-2-13b-hf-BC5CDR-chemical)  | [PMC-LLAMA 13B](https://huggingface.co/clinicalnlplab/finetuned-PMCLLaMA-13B-BC5CDR-chemical)|
| [NER]NCBI Disease        | [LLAMA 2 13B](https://huggingface.co/clinicalnlplab/finetuned-Llama-2-13b-hf-NCBI-disease)  | [PMC-LLAMA 13B](https://huggingface.co/clinicalnlplab/finetuned-PMCLLaMA-13B-NCBI-disease)|
| [RE]ChemProt             | [LLAMA 2 13B](https://huggingface.co/clinicalnlplab/finetuned-Llama-2-13b-hf-chemprot)  | [PMC-LLAMA 13B](https://huggingface.co/clinicalnlplab/finetuned-PMCLLaMA-13B-chemprot)|
| [RE]DDI2013              | [LLAMA 2 13B](https://huggingface.co/clinicalnlplab/finetuned-Llama-2-13b-hf-DDI2013_train) | [PMC-LLAMA 13B](https://huggingface.co/clinicalnlplab/finetuned-PMCLLaMA-13B-DDI2013_train)|
| [MLC]HoC                 | [LLAMA 2 13B](https://huggingface.co/clinicalnlplab/finetuned-Llama-2-13b-hf-HoC)|[PMC-LLAMA 13B](https://huggingface.co/clinicalnlplab/finetuned-PMCLLaMA-13B-HoC)|
| [MLC]LitCovid            | [LLAMA 2 13B](https://huggingface.co/clinicalnlplab/finetuned-Llama-2-13b-hf-LitCovid)|[PMC-LLAMA 13B](https://huggingface.co/clinicalnlplab/finetuned-PMCLLaMA-13b-LitCovid)|
| [QA]MedQA(5-option)      | [LLAMA 2 13B](https://huggingface.co/clinicalnlplab/finetuned-Llama-2-13b-hf-MedQA)|[PMC-LLAMA 13B](https://huggingface.co/clinicalnlplab/finetuned-PMCLLaMA-13B-MedQA)|
| [QA]PubMedQA             | [LLAMA 2 13B](https://huggingface.co/clinicalnlplab/finetuned-Llama-2-13b-hf-PubmedQA)|[PMC-LLAMA 13B](https://huggingface.co/clinicalnlplab/finetuned-PMCLLaMA-13B-PubmedQA)|
| [Summarization]PubMed    | [LLAMA 2 13B](https://huggingface.co/clinicalnlplab/finetuned-LLaMA-2-13b-hf-PubmedSumm)||[PMC-LLAMA 13B](https://huggingface.co/clinicalnlplab/finetuned-PMCLLaMA-PubmedSumm)|
| [Summarization]MS^2      | [LLAMA 2 13B](https://huggingface.co/clinicalnlplab/finetuned-Llama-2-13b-hf-MS2)|[PMC-LLAMA 13B](https://huggingface.co/clinicalnlplab/finetuned-PMCLLaMA-13B-MS2)|
| [Simplification]Cochrane | [LLAMA 2 13B](https://huggingface.co/clinicalnlplab/finetuned-Llama-2-13b-hf-CochranePLS)|[PMC-LLAMA 13B](https://huggingface.co/clinicalnlplab/finetuned-PMCLLaMA-13B-CochranePLS)|
| [Simplification]PLOS     | [LLAMA 2 13B](https://huggingface.co/clinicalnlplab/finetuned-Llama-2-13b-hf-PLOS)|[PMC-LLAMA 13B](https://huggingface.co/clinicalnlplab/finetuned-PMCLLaMA-PLOS)|

## 2. Inference

### 2.1 Inference for GPT models

The inference codes for GPT models are under [the GPT folder](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/tree/main/GPT)

To generate predictions for the generative/reasoning tasks (**[QA]MedQA(5-option)**, **[QA]PubMedQA**, **[Summarization]PubMed**, **[Summarization]MS^2**, **[Simplification]Cochrane**, **[Simplification]PLOS**), please use the following command:

```bash
python generative_tasks/run_gpt.py \
 --dataset {medqa5 | pubmedqa | pubmed | ms2 | cochrane | plos} \
 --model {gpt-35-turbo-16k | gpt-4-32k } \
 --setting {zero_shot | one_shot}
```
Predictions and corresponding gold labels are saved in JSON format, for example, `ms2_gpt-4-32k_one_shot.json`. The JSON files include both the predicted outputs and the gold standard labels for all examples within this dataset.

To generate predictions for the extractive/classification tasks (**[NER]BC5CDR-chemical**, **[NER]NCBI Disease**, **[RE]ChemProt**, **[RE]DDI2013**, **[MLC]HoC**,  **[MLC]LitCovid**), please use the following command:

```bash
python extractive_tasks/run_gpt.py
```
and
```bash
python extractive_tasks/run_convert_pred_2_json.py
```
to generate all predictions (6 extractive tasks for GPT-3.5 / 4, zero_shot / one_shot) all together. Predictions and corresponding gold labels are saved in JSON format, for example, `Hoc_gpt4_os.json`. The JSON files include both the predicted outputs and the gold standard labels for all examples within this dataset.

## 2.2 Inference for Llama models

The inference codes for GPT models are under [the llama folder]([https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/tree/main/GPT](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/tree/main/llama))

Please adhere to the instructions in the `llama` folder. Note that the evaluation script within this folder serves merely as a reference. For consistent results across all models — including Llama models and GPT models — we used `run_eval.py` for evaluations.

Predictions and corresponding gold labels are saved in JSON format, for example, `ms2_llama2_13b_chat_one_shot.json`. The JSON files include both the predicted outputs and the gold standard labels for all examples within this dataset.

## 3. Instruction fine-tuning Llama models

The instruction fine-tuning codes are under [the llmindcarft folder](https://github.com/BIDS-Xu-Lab/LLMindcraft/tree/ba60e8f862024067dcc78311dfcab144ef648bf2).

Please adhere to the instructions in the folder, which provides both the preprocessing scripts and fine-tuning docker images. 

For NER and RE tasks, run:
```bash
./llama/scripts/run-NER.sh 
```
or
```bash
./llama/scripts/run-RE.sh 
```
The models arguement could be set to any huggingface-based LLaMA models.



## 4. Evaluation

Please use [run_eval.py](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/blob/main/run_eval.py) for evaluation.

Before evaluation, please download [BART checkpoint](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) (for BART metrics evaluation).

To evaluate on various datasets or tasks, please use the following command:
```bash
python run_eval.py \
 --json_file {ms2_gpt-4-32k_one_shot.json | ms2_llama2_13b_chat_one_shot.json | ...} \
 --format_type {gpt | llama} \ 
 --task {NER | RE | MLC | QA | summarization | simplification}
```


## 5. Results

|             | Main metrics |SOTA results before LLMs | GPT-3.5 0s | GPT-4 0s  | LLAMA2 13B 0s | GPT-3.5 1s | GPT-4 1s  | LLAMA2 13B 1s | GPT-3.5 5s | GPT-4 5s  | LLAMA2 13B 5s | LLAMA2 13B fine-tuned | PMC LLAMA 13B fine-tuned |
|-------------|-----|---------------------|---------|---------|------------|---------|---------|------------|------------|---------------|------------|------------|---------------|
| [NER]BC5CDR-chemical     | Entity F1       | 0.9500  | 0.6274  | 0.7993 | 0.3944  | 0.7133  | **0.8327***  |0.6276 | 0.7228 | 0.7979 | 0.5530 | **0.9149** | 0.9063 |
| [NER]NCBI Disease        | Entity F1       | 0.9090  | 0.4060  | 0.5827 | 0.2211  | 0.4817  | 0.5988  |0.3811 | 0.4309 | **0.6389*** | 0.4847 | **0.8682*** | 0.8353 |
| [RE]ChemProt             | Macro F1        | 0.7344  | 0.1345  | 0.3250 | 0.1392  | 0.1280  | 0.3391  |0.0718 | 0.1758 | **0.3756** | 0.0967 | **0.4612*** | 0.3111 |
| [RE]DDI2013              | Macro F1        | 0.7919  | 0.2004  | 0.2968 | 0.1305  | 0.2126  | **0.3312**  |0.1779 | 0.1706 | 0.3276 | 0.1663 | **0.6218** | 0.5700 |
| [MLC]HoC                 | Macro F1        | 0.8882  | 0.6722  | **0.7109** | 0.1285  | 0.6671  | 0.7093  |0.3072 | 0.6994 | 0.7099 | 0.1797 | **0.6957*** | 0.4221 |
| [MLC]LitCovid            | Macro F1        | 0.8921  | 0.5967  | 0.5883 | 0.3825  | 0.6009  | 0.5901  |0.4808 | **0.6179** | 0.6077 | 0.3305 | **0.5725*** | 0.4273 |
| [QA]MedQA(5-option)      | Accuracy        | 0.4195  | 0.4988  | 0.7156 | 0.2522  | 0.5161  | 0.7439  |0.2899 | 0.5208 | **0.7651*** | 0.3504 | **0.4462*** | 0.3975 |
| [QA]PubMedQA             | Accuracy        | 0.7340  | 0.6560  | 0.6280 | 0.5520  | 0.4600  | 0.7100  |0.2660 | 0.6920 | **0.7580*** | 0.6000 | **0.8040*** | 0.7680 |
| [Summarization]PubMed    | Rouge-L         | 0.4316  | 0.2274  | 0.2419 | 0.1190  | 0.2351  | 0.2427  |0.0989 | 0.2423 | **0.2444** | 0.1629 | **0.1857*** | 0.1684 |
| [Summarization]MS^2      | Rouge-L         | 0.2080  | 0.0889  | 0.1224 | 0.0948  | 0.1132  | **0.1248**  |0.0320 | 0.1013 | 0.1218 | 0.1205 | **0.0934*** | 0.0059 |
| [Simplification]Cochrane | Rouge-L         | 0.4476  | 0.2365  | 0.2375 | 0.2081  | 0.2447  | 0.2385  |0.2207 | **0.2470** | 0.2469 | 0.2283 | 0.2355 | **0.2370** |
| [Simplification]PLOS     | Rouge-L         | 0.4368  | 0.2323  | 0.2253 | 0.2121  | **0.2449***  | 0.2386  |0.1836 | 0.2416 | 0.2409 | 0.1656 | **0.2583** | 0.2577 |
|                          | Macro-average   | 0.6536  | 0.3814  | 0.4561 | 0.2362  | 0.3848  | 0.4750  |0.2614 | 0.4052 | **0.4862** | 0.2866 | **0.5131** | 0.4422 |


## 5. Additional results
Additional results are under [Supplementary_Materials](https://github.com/BIDS-Xu-Lab/Biomedical-NLP-Benchmarks/tree/main/supplementary_materials).

## 6. Citation

If you use our work, please cite:

Chen, Q., Du, J., Hu, Y., Keloth, V.K., Peng, X., Raja, K., Zhang, R., Lu, Z. and Xu, H., 2023. [Large language models in biomedical natural language processing: benchmarks, baselines, and recommendations](https://arxiv.org/pdf/2305.16326). arXiv preprint arXiv:2305.16326.
