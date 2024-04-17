from tqdm import tqdm

import os
import openai
from openai import OpenAI
import random
import json
import argparse
import pandas as pd
import numpy as np

from utils.util import parse_to_int
from utils.prompt_utils import truncate_to_token_limit, GROUNDED_EVAL_PROMPT_TEMPLATE



def parse_to_int(value):
    try:
        # First, convert to float, then to int
        return int(float(value))
    except ValueError:
        # Handle the case where conversion fails
        print(f"Warning: Could not convert '{value}' to int.")
        return None


def truncate_to_token_limit(prompt, max_tokens=12000):
    """
    Truncate the input prompt to ensure it is within the maximum token limit.
    """
    # Average length of one token is roughly 4 characters for English
    avg_token_size = 4
    max_chars = max_tokens * avg_token_size

    # Truncate the tokens if necessary
    if len(prompt) > max_chars:
        # Truncate and keep the most recent tokens
        truncated_prompt = prompt[-max_chars:]
        print(f"original length: {len(prompt)}")
        print(f"truncated length: {len(truncated_prompt)}")
        return truncated_prompt
    else:
        return prompt

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--pattern", type=str, choices=["merged", "adapter", "plain"])
parser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106")
parser.add_argument("--num_samples", type=int, default=-1)
parser.add_argument("--num_partitions", type=int, default=4)
parser.add_argument("--partition", type=int, default=-1, choices=[-1, 0, 1, 2, 3])
parser.add_argument("--split", type=str, default="full", choices=["train", "test", "full"])
parser.add_argument("--sanity_check", action=argparse.BooleanOptionalAction)
parser.add_argument("--all_questions", action=argparse.BooleanOptionalAction)
args = parser.parse_args()


model_name = args.model


openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

model = model_name
temperature = 0.7
max_tokens = 512
top_p = 0.8
seed = 1234


df = pd.read_csv(args.data_file)

if not args.all_questions:
    # Group by 'cluster_id' and select the first row from each group
    df = df.groupby('cluster_id').first().reset_index()


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
    
df["model_resp"] = ""

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        df_line = df.loc[idx]
        user_message = GROUNDED_EVAL_PROMPT_TEMPLATE.format(df_line["question"])
        
        # zero shot inference without in-context examples
        messages = [{"role": "user", "content": user_message}]
        if args.sanity_check:
            print(user_message)
            print()
        
        num_retries = 10
        
        for _ in range(num_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                response_content = response.choices[0].message.content.strip()
                prompt_tokens = response.usage.prompt_tokens
                
                df.at[idx, "model_resp"] = response_content
                break
            except Exception as e:
                print(e)
                print(f"error generating output at cluster {df_line['cluster_id']}, retrying...")
    except Exception as e:
        print(e)
        print(f"error encountered at cluster {idx}, continuing...")
        continue
    
df.to_csv(args.output_file, index=None)