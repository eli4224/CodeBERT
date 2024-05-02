# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import Model, get_reinitialized_roberta, ModelPositional, ModelMidAttention
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)

from tqdm import tqdm, trange
import multiprocessing

cpu_cont = 16

from parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 url,
                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.url = url


@torch.no_grad()
def mask_tokens(tokens, tokenizer, mask_prob=0.15):
    mask_id = tokenizer.mask_token_id  # Get the integer ID for the [MASK] token
    padding_id = tokenizer.pad_token_id
    token_shape = tokens.shape
    tokens = tokens.view(-1)
    non_pad_tokens_idx = torch.argwhere(tokens != padding_id).view(-1)
    num_to_mask = int(len(non_pad_tokens_idx) * mask_prob)
    masking_indices = np.random.choice(non_pad_tokens_idx, num_to_mask, replace=False)

    masked_tokens = tokens.clone().detach()
    for idx in masking_indices:
        random_prob = np.random.rand()
        if random_prob < 0.8:
            # 80% replace with [MASK]
            masked_tokens[idx] = mask_id
        elif random_prob < 0.9:
            # 10% replace with a random word
            masked_tokens[idx] = np.random.randint(len(tokenizer))
        # 10% unchanged
    return masked_tokens.view(token_shape)


def convert_examples_to_features(item):
    js, tokenizer, args = item
    # code
    parser = parsers[args.lang]
    # extract data flow
    code_tokens, dfg = extract_dataflow(js['original_string'], parser, args.lang)
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]
    # truncating
    # code_tokens = code_tokens[:args.code_length + args.data_flow_length - 3 - min(len(dfg), args.data_flow_length)]
    code_tokens = code_tokens[:args.code_length]

    # nl stuff
    nl = ' '.join(js['docstring_tokens'])
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 3]

    text_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
    text_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(text_tokens))]

    dfg = dfg[:args.code_length + args.nl_length + args.data_flow_length - len(code_tokens) - len(nl_tokens) - 3]
    text_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    text_ids += [tokenizer.unk_token_id for x in dfg]

    padding_length = args.code_length + args.data_flow_length + args.nl_length - len(text_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    text_ids += [tokenizer.pad_token_id] * padding_length

    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]


    return InputFeatures(text_tokens, text_ids, position_idx, dfg_to_code, dfg_to_dfg, js['url'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None, ignore_cache=False):
        self.args = args
        prefix = file_path.split('/')[-1][:-6]
        cache_file = args.output_dir + '/' + prefix + '.pkl'
        self.tokenizer = tokenizer
        if os.path.exists(cache_file) and not ignore_cache:
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            self.examples = []
            data = []
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    data.append((js, tokenizer, args))
            # self.examples = [convert_examples_to_features(item) for item in data]
            self.examples = pool.map(convert_examples_to_features, tqdm(data, total=len(data)))
            pickle.dump(self.examples, open(cache_file, 'wb'))

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length + self.args.nl_length,
                              self.args.code_length + self.args.data_flow_length + self.args.nl_length), dtype=bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].code_ids):
            if i in [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id, self.tokenizer.cls_token_id, self.tokenizer.mask_token_id]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        try:
            for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
                for a in nodes:
                    if a + node_index < len(self.examples[item].position_idx):
                        attn_mask[idx + node_index, a + node_index] = True
        except Exception as e:
            print("hello cowan")

        return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer, pool):
    """ Train the model """
    # get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file, pool, ignore_cache=False)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    # get optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * args.num_train_epochs)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader) * args.num_train_epochs)

    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    model.train()
    model.to(args.device)
    tr_num, tr_loss, best_acc = 0, 0, 0
    all_steps = 0
    for idx in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # get inputs
            labels = batch[0].to(args.device)
            code_inputs = mask_tokens(batch[0], tokenizer).to(args.device)
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            # get code and nl vectors
            code_vec = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)

            # calculate scores and loss
            loss_fct = CrossEntropyLoss()
            with torch.no_grad():
                loss_mask = (labels != code_inputs)
            masked_code_vec = code_vec[loss_mask]
            masked_labels = labels[loss_mask]
            labels_one_hot = torch.nn.functional.one_hot(masked_labels, num_classes=len(tokenizer)).float()
            loss = loss_fct(masked_code_vec, labels_one_hot)

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % 100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step + 1, round(tr_loss / tr_num, 5)))
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if all_steps % 1000 == 0: # evaluate
                results = evaluate(args, model, tokenizer, args.eval_data_file, pool, eval_when_training=True)
                for key, value in results.items():
                    logger.info("  %s = %s", key, value)

                    # save best model
                if results['eval_accuracy'] > best_acc:
                    best_acc = results['eval_accuracy']
                    logger.info("  " + "*" * 20)
                    logger.info("  Best accuracy:%s", best_acc.item())
                    logger.info("  " + "*" * 20)

                    checkpoint_prefix = 'checkpoint-best-acc'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                    torch.save(model_to_save.state_dict(), output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

                model.train()

            all_steps += 1


def evaluate(args, model, tokenizer, file_name, pool, eval_when_training=False):
    # get training dataset
    test_dataset = TextDataset(tokenizer, args, file_name, pool, ignore_cache=False)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=4)


    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    model.to(args.device)
    tr_num, tr_loss, best_mrr = 0, 0, 0
    total_correct = 0
    for step, batch in enumerate(test_dataloader):
        # get inputs
        labels = batch[0].to(args.device)
        code_inputs = mask_tokens(batch[0], tokenizer).to(args.device)
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        # get code and nl vectors
        code_vec = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)

        # calculate scores and loss
        loss_fct = CrossEntropyLoss()
        with torch.no_grad():
            loss_mask = (labels != code_inputs)
        masked_code_vec = code_vec[loss_mask]
        masked_labels = labels[loss_mask]
        labels_one_hot = torch.nn.functional.one_hot(masked_labels, num_classes=len(tokenizer)).float()
        loss = loss_fct(masked_code_vec, labels_one_hot)
        total_correct = total_correct + torch.sum(torch.argmax(masked_code_vec, dim=-1) == masked_labels)

        # report loss
        tr_loss += loss.item()
        tr_num += 1

    result = {
        "eval_loss": tr_loss / tr_num,
        "eval_accuracy": total_correct / tr_num
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")

    parser.add_argument("--lang", default=None, type=str,
                        help="language.")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--positional', type=str, default="constant",
                        help="The default positional encoding (constant), a "
                             "random walk node2vec encoding (random_walk), or "
                             "a laplacian encoding (laplacian).",
                        choices=["constant", "random_walk", "laplacian"])

    pool = multiprocessing.Pool(cpu_cont)

    # print arguments
    args = parser.parse_args()

    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # build model
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    positional_size = 36
    # When running in standard mode, set this to 0. When running in Concatenation mode, set this to 36.
    # When using constant positional encoding, this variable isn't used.
    extra_positional_size = 36

    if args.positional in ["random_walk", "laplacian"]:
        # config.hidden_size = config.hidden_size + extra_positional_size
        model = get_reinitialized_roberta(positional_size, extra_positional_size)
        model = ModelPositional(model, tokenizer, positional_size, extra_positional_size, args.positional)
    elif args.positional == "constant":
        config = RobertaConfig.from_pretrained(args.config_name) if args.config_name else None
        # Use this line to load the preprocessed model:
        # model = RobertaModel.from_pretrained(args.model_name_or_path)
        # Use this line to load the reinitialized original model:
        model = Model(get_reinitialized_roberta(config))
        # Use this line to use the Delayed Full Attention model:
        # model = ModelMidAttention(config if config is not None else RobertaConfig())
    else:
        raise ValueError(f"Invalid positional encoding type: {args.positional}")
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)

    # Training
    if args.do_train:
        train(args, model, tokenizer, pool)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir), strict=False)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.eval_data_file, pool)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir), strict=False)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.test_data_file, pool)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    return results


if __name__ == "__main__":
    main()


