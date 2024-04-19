import pandas as pd
from tqdm import tqdm
import argparse
import os
import math
import openai
from openai import OpenAI

from utils.prompt_utils import KNOWLEDGE_ENTAIL_SYSTEM_PROMPT, KNOWLEDGE_ENTAIL_USER_TEMPLATE


"""
Example Usage:

python evaluation/knowledge_entailment.py --data_file <path_to_grounded_eval_output> --output_file <output_path> --model_name <model_under_evaluation>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--knowledge_file", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--aug", action=argparse.BooleanOptionalAction)
    parser.add_argument("--sanity_check", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()

    df = pd.read_csv(args.data_file)
    knowledge = pd.read_csv(args.knowledge_file)

    if args.sanity_check:
        df = df.head(5)

    if args.aug:
        df["knowledge_entailment_aug"] = 0.0
    else:
        df["knowledge_entailment"] = 0.0
        
    model = "gpt-4-1106-preview"
    temperature = 0
    max_tokens = 1
    top_p = 0.01
    seed = 1234

    num_retries = 10
    for idx, _ in tqdm(df.iterrows(), total=len(df)):
        for _ in range(num_retries):
            try:
                df_line = df.iloc[idx]
                knowledge_line = knowledge.loc[knowledge['cluster_id'] == df_line['cluster_id']].iloc[0]
                if args.aug:
                    desc, model_resp = knowledge_line["desc"], df_line["model_resp_aug"]
                else:
                    desc, model_resp = knowledge_line["desc"], df_line["model_resp"]
                messages = [
                    {"role": "system", "content": KNOWLEDGE_ENTAIL_SYSTEM_PROMPT}, 
                    {"role": "user", "content": KNOWLEDGE_ENTAIL_USER_TEMPLATE.format(model_resp, desc)},
                ]
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    logprobs=True,
                    top_logprobs=5,
                )
                
                top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                
                if args.sanity_check:
                    print(top_logprobs)
                
                yes_prob, no_prob = 0, 0
                for logprob in top_logprobs:
                    if logprob.token == "Yes":
                        yes_prob = math.exp(logprob.logprob)
                    elif logprob.token == "No":
                        no_prob = math.exp(logprob.logprob)
                
                if yes_prob > 0 or no_prob > 0:
                    if args.aug:
                        df.at[idx, "knowledge_entailment_aug"] = yes_prob
                    else:
                        df.at[idx, "knowledge_entailment"] = yes_prob
                else:
                    print('Warning: the probabilities for both "Yes" and "No" are 0, continuing...')
                    # the default value for knowledge entailment is 0
                
                break
            except Exception as e:
                print(f'encountered error at row {idx}: {e}')
                print("retrying...")
                continue

    if args.aug:
        print(f"Average score for {args.model_name}: {df.loc[:, 'knowledge_entailment_aug'].mean()}")
    else:
        print(f"Average score for {args.model_name}: {df.loc[:, 'knowledge_entailment'].mean()}")
    df.to_csv(args.output_file, index=None)

if __name__ == "__main__":
    main()