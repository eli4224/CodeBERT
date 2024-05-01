

# GraphCodeBERT Pretraining
This repository attempts to re-train GraphCodeBERT model from the paper [GraphCodeBERT: Pre-training Code Representations with Data Flow](https://arxiv.org/abs/2009.08366). GraphCodeBERT is a pre-trained model for code representation learning, which is pre-trained on a large-scale code corpus with data flow graphs.

The Microsoft did not release the code used to pretrain the model. This repository currently has reimplimented the masking language model pretraining task and, as a demonstration of the pretraining capabilities, new kinds of graph positional encodings for the model. The code is adapted from the code and data of the Code Search finetuning task released by Microsoft alongside the paper.
## Data Preprocess

Running the model requires the same data preprocessing as the Code Search task. This is copy and pasted below for convenience:

Download and preprocess data using the following command.
```shell
unzip dataset.zip
cd dataset
bash run.sh 
cd ..
```

## Dependency 

- pip install torch
- pip install transformers
- pip install tree_sitter

### Tree-sitter (optional)

If the built file "parser/my-languages.so" doesn't work for you, please rebuild as the following command:

```shell
cd parser
bash build.sh
cd ..
```

## Pre-train

We fine-tuned the model on 2*V100-16G GPUs. 
```shell
lang=ruby
mkdir -p ./saved_models/$lang
python run.py \
    --output_dir=./saved_models/$lang \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_train \
    --train_data_file=dataset/$lang/train.jsonl \
    --eval_data_file=dataset/$lang/valid.jsonl \
    --test_data_file=dataset/$lang/test.jsonl \
    --codebase_file=dataset/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1 \
    --positional constant | tee saved_models/$lang/train.log
```
## Inference and Evaluation

```shell
lang=ruby
python run.py \
    --output_dir=./saved_models/$lang \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_eval \
    --do_test \
    --train_data_file=dataset/$lang/train.jsonl \
    --eval_data_file=dataset/$lang/valid.jsonl \
    --test_data_file=dataset/$lang/test.jsonl \
    --codebase_file=dataset/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee saved_models/$lang/test.log
    --positional constant | tee saved_models/$lang/train.log
```

This code was initially run on 1 P100 GPU, but should be able to run on additional GPUs, which is highly recommended.