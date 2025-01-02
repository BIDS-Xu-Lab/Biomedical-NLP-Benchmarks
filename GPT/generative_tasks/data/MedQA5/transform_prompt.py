__author__ = "qiao"

"""
transform the style of the prompts
"""

import json

if __name__ == "__main__":

	for split in ["train", "test"]:

		data = json.load(open(f"MedQA_5Opt_{split}.json"))

		for entry in data:
			
			for prompt in ["sentence", "gold"]:
				

				for choice in ["A", "B", "C", "D", "E"]:

					entry[prompt] = entry[prompt].replace(f"{choice}:", f"{choice}. ")


				entry[prompt] = entry[prompt].replace("A.", "\nA.").replace("\n ", "\n").strip()

		with open(f"medqa5_{split}.json", "w") as f:
			json.dump(data, f, indent=4)
