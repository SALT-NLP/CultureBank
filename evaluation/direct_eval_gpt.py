from tqdm import tqdm
from sklearn.metrics import f1_score

import os
import re
import openai
from openai import OpenAI
import random
import json
import argparse
import pandas as pd
import numpy as np

from utils.constants import EVAL_FIELDS
from utils.prompt_utils import FIELD_DEFINITIONS, DIRECT_EVAL_PROMPT_TEMPLATE
from utils.util import extract_yes_or_no


parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106")
parser.add_argument("--num_samples", type=int, default=-1)
parser.add_argument("--num_partitions", type=int, default=4)
parser.add_argument("--partition", type=int, default=-1, choices=[-1, 0, 1, 2, 3])
parser.add_argument("--split", type=str, default="full", choices=["train", "test", "full"])
parser.add_argument("--sanity_check", action=argparse.BooleanOptionalAction)
args = parser.parse_args()


model_name = args.model


openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

model = model_name
temperature = 0.7
max_tokens = 5
top_p = 0.01
seed = 1234


df = pd.read_csv(args.data_file)


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
    


# bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# labels = [1, 2, 3, 4, 5]
df = df.dropna(subset=['norm'])
df["model_resp"] = ""
# df["model_resp"] = 0
# df['norm_level'] = pd.cut(df['norm'], bins=bins, labels=labels, include_lowest=True, right=True)
# df['norm_level'] = df['norm_level'].astype(int)

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
        messages = [{"role": "user", "content": user_message}]
        if args.sanity_check:
            print(user_message)
            print()
        # print(prompt)
        
        num_retries = 1
        
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
                
                if args.sanity_check:
                    print(response_content)
                    print()
                
                parsed_resp = extract_yes_or_no(response_content)
                df.at[idx, "model_resp"] = parsed_resp
                
                pred = parsed_resp == "Yes"
                target = df_line['norm'] > 0.5
                if pred == target:
                    num_correct += 1
                break
            except Exception as e:
                print(e)
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