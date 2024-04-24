__author__ = "qiao"

from datasets import load_dataset
import json
import os
import pandas as pd
import sys
import time
import tiktoken
from templates_medprompt import templates

import openai
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = "2023-07-01-preview"
openai.api_key = ""

if __name__ == "__main__":
	# args
	dataset = sys.argv[1]
	model = sys.argv[2]
	
	# controlling the max exemplar length
	if model == "gpt-35-turbo-16k":
		input_length = 14000
	elif model == "gpt-4-32k":
		input_length = 30000
	
	print("Running", dataset)

	# medprompt_1, medprompt_2, medprompt_5
	setting = sys.argv[3]
	N = int(setting.split("_")[-1])

	# for truncating long prompts
	encoding = tiktoken.encoding_for_model(model)
	
	# first loading the dataset
	if dataset == "pubmedqa":
		train_path = "zhizheng_data/PubMedQA/pubmedqa_train.json"
		test_path = "zhizheng_data/PubMedQA/pubmedqa_test.json"
		sim_path = "zhizheng_data/PubMedQA/similarity.rank.json"
	elif dataset == "ms2":
		train_path = "zhizheng_data/MS2/ms2_train.json"
		test_path = "zhizheng_data/MS2/ms2_test.json"
		sim_path = "zhizheng_data/MS2/similarity.rank.json"
	elif dataset == "pubmed":
		train_path = "zhizheng_data/PubMed/pubmed_train.json" 
		test_path = "zhizheng_data/PubMed/pubmed_test.json"
		sim_path = "zhizheng_data/PubMed/similarity.rank.json"
	elif dataset == "cochrane":
		train_path = "zhizheng_data/Cochrane/cochrane_train.json"
		test_path = "zhizheng_data/Cochrane/cochrane_test.json"
		sim_path = "zhizheng_data/Cochrane/similarity.rank.json"
	elif dataset == "medqa":
		train_path = "zhizheng_data/MedQA/medqa_train.json"
		test_path = "zhizheng_data/MedQA/medqa_test.json"
		sim_path = "zhizheng_data/MedQA/similarity.rank.json"
	elif dataset == "medqa5":
		train_path = "zhizheng_data/MedQA5/medqa5_train.json"
		test_path = "zhizheng_data/MedQA5/medqa5_test.json"
		sim_path = "zhizheng_data/MedQA5/similarity.rank.json"
	elif dataset == "plos":
		train_path = "zhizheng_data/PLOS/plos_train.json"
		test_path = "zhizheng_data/PLOS/plos_test.json"
		sim_path = "zhizheng_data/PLOS/similarity.rank.json"

	train = json.load(open(train_path))
	idx2train = {str(entry["index"]): entry for entry in train}
	test = json.load(open(test_path))
	sim = json.load(open(sim_path))
	sim = {list(entry.items())[0][0] : list(entry.items())[0][1] for entry in sim}
	
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

	#for idx, row in data.iterrows():
	for idx, entry in enumerate(test):
		# start when idx == truth due to sequential caching
		if idx < len(truth):
			continue

		index = str(entry["index"])
		instance, answer = entry["sentence"], entry["gold"]
		
		# getting the exemplars
		neighbors = sim[index]
		examples = ""
		for neighbor in neighbors[:N]:
			neighbor = idx2train[neighbor["train_index"]]
			examples += f"INPUT: {neighbor['sentence']}\n"
			examples += f"OUTPUT: {neighbor['gold']}\n\n"

		examples_tokens = encoding.encode(examples)
		instance_tokens = encoding.encode(instance)
		
		raw_input_tokens = len(examples_tokens) + len(instance_tokens)

		# only truncate if the sum length is longer than the max input length
		if raw_input_tokens > input_length:
			# how many tokens to cut
			to_cut = raw_input_tokens - input_length
			
			# if to cut more than all examples tokens, need to also cut instance tokens
			if to_cut > len(examples_tokens):
				examples = ""
				instance = encoding.decode(instance_tokens[:-(to_cut-len(examples_tokens))])

			# if not, just cut the example tokens
			else:
				examples = encoding.decode(examples_tokens[:-to_cut])

		prompt = templates[dataset]["medprompt"](examples, instance)
		prompts.append(prompt)
		
		time.sleep(2)

		try:
			response = openai.ChatCompletion.create(
				engine=model,
				messages=[{"role": "user", "content": prompt}],
				temperature=0,
				max_tokens=2000,
				request_timeout=600,
			)
		
			if "content" not in response["choices"][0]["message"]:
				result = ""
			else:
				result = response["choices"][0]["message"]["content"]

		except openai.error.InvalidRequestError as E:
			if "filtered " in repr(E):
				result = ""	
			else:
				break

		truth.append(answer)
		preds.append(result)

		with open(output_path, "w") as f:
			output = {
				"truth": truth,
				"preds": preds,
				"prompts": prompts,
			}

			json.dump(output, f, indent=4)
