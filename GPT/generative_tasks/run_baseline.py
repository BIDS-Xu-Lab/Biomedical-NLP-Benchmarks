__author__ = "qiao"

from datasets import load_dataset
import json
import os
import pandas as pd
import sys
import time
import tiktoken
from templates import templates

import openai
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = "2023-07-01-preview"
openai.api_key = ""

if __name__ == "__main__":
	# args
	dataset = sys.argv[1]
	model = sys.argv[2]
	setting = sys.argv[3]

	# for truncating long prompts
	encoding = tiktoken.encoding_for_model(model)

	if dataset == "pubmedqa":
		data = pd.read_csv("PubMedQA/datasets/full_set/test.tsv", sep="\t")
	elif dataset == "biosses":
		# to be replaced by MedQA (https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
		data = pd.read_csv("BIOSSES/datasets/full_set/test.tsv", sep="\t")
	elif dataset == "ms2":
		data = load_dataset("allenai/mslr2022", "ms2", split="validation", cache_dir="./qiao_datasets")
		data = pd.DataFrame(data)
	elif dataset == "pubmed":
		data = load_dataset("ccdv/pubmed-summarization", "document", split="test", cache_dir="./qiao_datasets")
		data = pd.DataFrame(data)

	elif dataset == "cochrane":
		data = load_dataset("GEM/cochrane-simplification", split="test", cache_dir="./qiao_datasets") 
		data = pd.DataFrame(data)
	elif dataset == "medqa":
		data = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", cache_dir="./qiao_datasets")	
		data = pd.DataFrame(data)

	elif dataset == "medqa5":
		data = pd.read_json('qiao_datasets/data_clean/questions/US/test.jsonl', lines=True)

	elif dataset == "plos" or dataset == "plos2":
		data = pd.read_json("PLOS Simplification/dataset/plos_corpus/test_plos.jsonl", lines=True)
	
	# loading the cached results (sequentially)
	output_path = f"qiao_results/{dataset}_{model}_{setting}.json"

	if os.path.exists(output_path):
		output = json.load(open(output_path))
		
		truth = output["truth"]
		preds = output["preds"]
		prompts = output["prompts"]

	else:
		truth = []
		preds = []
		prompts = []

	for idx, row in data.iterrows():

		# start when idx == truth
		if idx < len(truth):
			continue

		if dataset == "pubmedqa":
			question, abstract, answer = row["QUESTION"], row["CONTEXTS"], row["final_decision"]
			prompt = templates["pubmedqa"][setting](question, abstract)

		elif dataset == "biosses":
			sent1, sent2, answer = row["sentence1"], row["sentence2"], row["score"]
			prompt = templates["biosses"][setting](sent1, sent2)

		elif dataset == "medqa":
			question, options, answer = row["question"], row["options"], row["answer_idx"]
			choices = ""
			for k, v in options.items():
				choices += f"{k}. {v}\n" 
			prompt = templates["medqa"][setting](question, choices)

		elif dataset == "medqa5":
			question = row["question"]
			choices = ""
			for k, v in row["options"].items():
				choices += f"{k}. {v}\n" 
			answer = row["answer_idx"]

			prompt = templates["medqa5"][setting](question, choices)

		elif dataset == "ms2":
			abstracts, answer = row["abstract"], row["target"]
			abstracts = " ".join(abstracts)

			tokens = encoding.encode(abstracts)
			
			if model == "gpt-35-turbo-16k":
				length = 15000
			elif model == "gpt-4-32k":
				length = 31000

			if setting == "one_shot":
				length -= 2500 
			
			abstracts = encoding.decode(tokens[:length])

			prompt = templates["ms2"][setting](abstracts)

		elif dataset == "pubmed":
			full_text, answer = row["article"], row["abstract"]

			tokens = encoding.encode(full_text)
			
			if model == "gpt-35-turbo-16k":
				length = 15000
			elif model == "gpt-4-32k":
				length = 31000

			if setting == "one_shot":
				length -= 2500 
			
			full_text = encoding.decode(tokens[:length])

			prompt = templates["pubmed"][setting](full_text)

		elif dataset == "cochrane":
			source, answer = row["source"], row["target"]
			prompt = templates["cochrane"][setting](source)

		elif dataset == "plos":
			article, answer = row["article"], row["plain language summary"]
			prompt = templates[dataset][setting](article)

		elif dataset == "plos2":
			article, answer = row["abstract"], row["plain language summary"]
			prompt = templates[dataset][setting](article)

		prompts.append(prompt)
		
		time.sleep(1)

		try:
			response = openai.ChatCompletion.create(
				engine=model,
				messages=[{"role": "user", "content": prompt}],
				temperature=0,
			)
		
			if "content" not in response["choices"][0]["message"]:
				result = ""
			else:
				result = response["choices"][0]["message"]["content"]

		except openai.error.InvalidRequestError:
			result = ""

		truth.append(answer)
		preds.append(result)

		with open(output_path, "w") as f:
			output = {
				"truth": truth,
				"preds": preds,
				"prompts": prompts,
			}

			json.dump(output, f, indent=4)
