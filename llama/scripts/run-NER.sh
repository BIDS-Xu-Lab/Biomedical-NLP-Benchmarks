#!/bin/bash

# Define the datasets and models as arrays
datasets=('[NER]NCBI_Disease' '[NER]BC5CDR_Chemical')
models=('axiong/PMC_LLaMA_13B')

# Loop over each dataset
for dataset in "${datasets[@]}"; do
    # Loop over each model
    for model in "${models[@]}"; do
        # Run the Python script with the current dataset and model
        CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 29503 llama/src/NER/multiple_GPU_train_V2-NER.py -d "$dataset" -m "$model"
    done
done
