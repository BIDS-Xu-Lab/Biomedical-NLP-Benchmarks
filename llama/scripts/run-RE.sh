#!/bin/bash

# Define the datasets and models as arrays
datasets=('hippocrates/DDI2013_train')
models=('hippocrates/fine-tuned_medllama70b')

# Loop over each dataset
for dataset in "${datasets[@]}"; do
    # Loop over each model
    for model in "${models[@]}"; do
        # Run the Python script with the current dataset and model
        CUDA_VISIBLE_DEVICES=1,3,5,6,7 accelerate launch --main_process_port 29501 src/multiple_GPU_train_V2-RE.py -d "$dataset" -m "$model"
    done
done
