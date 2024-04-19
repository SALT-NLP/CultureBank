import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import json
import argparse
import openai
from openai import OpenAI

from utils.prompt_utils import FIELD_DEFINITIONS_SUMMARIZED, INCONTEXT_DESC, INCONTEXT_EXP, DESC_SYSTEM_PROMPT, DESC_USER_TEMPLATE
from utils.constants import DESC_FIELDS

"""
example usage: python evaluation/convert_to_desc.py
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--sanity_check", action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_partitions", type=int, default=4)
    parser.add_argument("--partition", type=int, default=-1)
    args = parser.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()

    df = pd.read_csv(args.data_file)

    if args.num_samples != -1:
        df = df.sample(n=args.num_samples, replace=False, random_state=1234)

    if args.partition != -1:
        assert args.partition < args.num_partitions
        partitions = np.array_split(df, args.num_partitions)

        for i in range(len(partitions)):
            print(f"partition {i}:")
            print(partitions[i].head())
            print()
        df = partitions[args.partition]

        print(f"currently processing {len(df)} clusters")
        print(df.head())

    model = "gpt-3.5-turbo-1106"
    engine = "chatgpt0613"
    temperature = 0.3
    max_tokens = 1000
    top_p = 0.2
    seed = 1234

    system_message = DESC_SYSTEM_PROMPT.format(json.dumps(FIELD_DEFINITIONS_SUMMARIZED, indent=4))
    incontext_user_message = DESC_USER_TEMPLATE.format(json.dumps(INCONTEXT_EXP, indent=4))
    incontext_assistant_message = INCONTEXT_DESC

    df_results = []

    for idx, _ in tqdm(df.iterrows(), total=len(df)):
        for _ in range(10):
            try:
                df_line = df.loc[idx]
                cultural_knowledge = {}
                for field in DESC_FIELDS:
                    cultural_knowledge[field] = df_line[field]
                
                user_message = DESC_USER_TEMPLATE.format(json.dumps(cultural_knowledge, indent=4))
                messages = [
                    {"role": "system", "content": system_message}, 
                    {"role": "user", "content": incontext_user_message},
                    {"role": "assistant", "content": incontext_assistant_message},
                    {"role": "user", "content": user_message},
                ]
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    # engine=engine,
                )
                response_content = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                description = response_content.strip()
                
                output_row = {}
                output_row["cluster_id"] = df_line["cluster_id"]
                output_row["desc"] = description

                df_results.append(output_row)
                break
            except Exception as e:
                print(f'encountered error at row {idx}: {e}')
                print("retrying...")
                continue

    df_results = pd.DataFrame.from_records(df_results, columns=["cluster_id", "desc"])
    df_results.to_csv(args.output_file, index=None)

if __name__ == "__main__":
    main()