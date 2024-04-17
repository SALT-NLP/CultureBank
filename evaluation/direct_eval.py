from transformers import AutoTokenizer
from tqdm import tqdm
from peft import PeftModel, AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import f1_score

import os
import re
import transformers
import torch
import random
import json
import argparse
import math
import pandas as pd
import numpy as np

from utils.constants import EVAL_FIELDS
from utils.prompt_utils import FIELD_DEFINITIONS, DIRECT_EVAL_PROMPT_TEMPLATE
from utils.util import extract_yes_or_no


def parse_to_int(value):
    try:
        # First, convert to float, then to int
        return int(float(value))
    except ValueError:
        # Handle the case where conversion fails
        print(f"Warning: Could not convert '{value}' to int.")
        return None




parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--pattern", type=str, default="plain", choices=["merged", "adapter", "plain", "awq"])
parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--adapters", default=[], nargs='+')
parser.add_argument("--num_samples", type=int, default=-1)
parser.add_argument("--num_partitions", type=int, default=4)
parser.add_argument("--threshold", type=int, default=-1)
parser.add_argument("--partition", type=int, default=-1, choices=[-1, 0, 1, 2, 3])
parser.add_argument("--split", type=str, default="full", choices=["train", "test", "full"])
parser.add_argument("--sanity_check", action=argparse.BooleanOptionalAction)
args = parser.parse_args()


model_name = args.model
tokenizer_path = args.tokenizer if args.tokenizer else model_name

transformers.set_seed(1234)

if args.pattern == "adapter":
    assert len(args.adapters) >= 1
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if len(args.adapters) == 1:
        # No need to merge
        text_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False
            ),
            attn_implementation="flash_attention_2",
        )
        pass
    elif len(args.adapters) > 1:
        # Need to merge
        text_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    print("----------------------------------------------------")
    print(f"Loaded the model {model_name}")

    if len(args.adapters) == 1:
        text_model = PeftModel.from_pretrained(text_model, args.adapters[0])
        
        print("--------------------NO MERGING----------------------")
        print(f"Loaded the adapter model {args.adapters[0]}")
    elif len(args.adapters) > 1:
        for adapter_name in args.adapters:
            text_model = PeftModel.from_pretrained(text_model, adapter_name)
            text_model = text_model.merge_and_unload()
    
            print("----------------------MERGING-----------------------")
            print(f"Loaded the adapter model {adapter_name}")
elif args.pattern == "merged" or args.pattern == "plain":
    text_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False
        ),
        attn_implementation="flash_attention_2",
    )
elif args.pattern == "awq":
    text_tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", device_map="auto")
else:
    raise NotImplementedError

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

df = pd.read_csv(args.data_file)

if args.threshold != -1:
    df = df.loc[df['norm_total'] >= args.threshold]
    print(f"a total of {len(df)} clusters with size >= {args.threshold}")
    americans = df.loc[df['representative_cultural group'] == 'American']
    print(f"total number of american clusters: {len(americans)}")
    unique_groups = df['representative_cultural group'].unique()
    print(f'total number of cultural groups: {len(unique_groups)}')

if args.split and args.split != "full":
    df = df.sample(frac=1, random_state=1234).reset_index(drop=True)
    train_split = 0.8
    if args.split == "train":
        df = df.head(int(len(df)*(train_split))).reset_index(drop=True)
    elif args.split == "test":
        test_split = (1.0 - train_split) / 2
        df = df.tail(int(len(df)*(test_split))).reset_index(drop=True)


if args.num_samples != -1:
    df = df.sample(n=args.num_samples, replace=False, random_state=1234)
elif args.partition != -1:
    assert args.partition < args.num_partitions
    partitions = np.array_split(df, args.num_partitions)

    for i in range(len(partitions)):
        print(f"partition {i}:")
        print(partitions[i].head())
        print()
    df = partitions[args.partition]

    print(f"currently processing {len(df)} clusters")
    print(df.head())


if args.sanity_check:
    df = df.head(5)

df = df.dropna(subset=['norm'])
df["model_resp"] = ""


num_correct = 0

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        df_line = df.loc[idx]
        cultural_knowledge = {}
        field_definitions = {}
        for field in EVAL_FIELDS:
            cultural_knowledge[field] = df_line[field]
            field_definitions[field] = FIELD_DEFINITIONS[field]
        
        user_message = DIRECT_EVAL_PROMPT_TEMPLATE.format(json.dumps(field_definitions, indent=4), json.dumps(cultural_knowledge, indent=4))
        
        # zero shot inference without in-context examples
        prompt = user_message
        prompt = truncate_to_token_limit(prompt)
        if args.sanity_check:
            print(prompt)
            print()
        
        num_retries = 1
        
        for _ in range(num_retries):
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = text_model.generate(**inputs, max_new_tokens=2, do_sample=True, temperature=0.8, top_k=10, top_p=0.01)
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if args.sanity_check:
                    print(output_text)
                    print()
                
                output_text = output_text[len(prompt) - len('<s>'):]
                
                if args.sanity_check:
                    print(output_text)
                    print()
                    
                parsed_resp = extract_yes_or_no(output_text)
                
                df.at[idx, "model_resp"] = parsed_resp
                
                pred = parsed_resp == "Yes"
                target = df_line['norm'] > 0.5
                if pred == target:
                    num_correct += 1
                
                break
            except Exception as e:
                print(e)
                print()
                print("generated output:")
                print(output_text)
                print(f"error generating output at cluster {df_line['cluster_id']}, retrying...")
    except Exception as e:
        print(e)
        print(f"error encountered at cluster {idx}, continuing...")
        continue
    
df.to_csv(args.output_file, index=None)

df = df.loc[df["model_resp"] != ""]
print(f"Direct evaluation results for {model_name}: {num_correct / len(df)}")


model_class = df["model_resp"]
target_class = df['norm'].apply(lambda x: "Yes" if x > 0.5 else "No")
weighted_f1 = f1_score(model_class, target_class, average='weighted')
print(f"weighted f1 score: {weighted_f1}")

macro_f1 = f1_score(model_class, target_class, average='macro')

print(f"macro f1 score: {macro_f1}")