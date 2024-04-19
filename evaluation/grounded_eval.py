from transformers import AutoTokenizer
from tqdm import tqdm
from peft import PeftModel, AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import os
import transformers
import torch
import random
import json
import argparse
import pandas as pd
import numpy as np

from utils.util import parse_to_int
from utils.prompt_utils import truncate_to_token_limit, GROUNDED_EVAL_PROMPT_TEMPLATE, GROUNDED_EVAL_PROMPT_AUG_TEMPLATE


"""
Example Usage:

python evaluation/benchmark.py --data_file <path_to_cultural_questions> --output_file <output_path> --pattern plain --model meta-llama/Llama-2-7b-chat-hf
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--pattern", type=str, choices=["merged", "adapter", "plain", "awq"])
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--adapters", default=[], nargs='+')
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--num_partitions", type=int, default=4)
    parser.add_argument("--partition", type=int, default=-1, choices=[-1, 0, 1, 2, 3])
    parser.add_argument("--split", type=str, default="full", choices=["train", "test", "full"])
    parser.add_argument("--aug", action=argparse.BooleanOptionalAction)
    parser.add_argument("--sanity_check", action=argparse.BooleanOptionalAction)
    parser.add_argument("--all_questions", action=argparse.BooleanOptionalAction)    # whether we benchmark on all questions or just select one question for each knowledge
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
            
            # text_model = AutoPeftModelForCausalLM.from_pretrained(args.adapters[0], device_map="cpu", torch_dtype=torch.bfloat16)
            # text_model = text_model.merge_and_unload()
            # merged_checkpoint = os.path.join(args.adapters[0], "final_merged_checkpoint")
            
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
    knowledge = pd.read_csv("/sphinx/u/culturebank/tiktok_results/evaluation/cultural_descriptions_v2.csv")

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

    if args.aug:
        df["model_resp_aug"] = ""
    else:
        df["model_resp"] = ""

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            df_line = df.loc[idx]
            if args.aug:
                knowledge_line = knowledge.loc[knowledge['cluster_id'] == df_line['cluster_id']].iloc[0]
                user_message = GROUNDED_EVAL_PROMPT_AUG_TEMPLATE.format(df_line["question"], knowledge_line['desc'])
            else:
                user_message = GROUNDED_EVAL_PROMPT_TEMPLATE.format(df_line["question"])
            
            # zero shot inference without in-context examples
            messages = [{"role": "user", "content": user_message}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = truncate_to_token_limit(prompt)
            if args.sanity_check:
                print(prompt)
                print()
            
            num_retries = 10
            
            for _ in range(num_retries):
                try:
                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                    outputs = text_model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=10, top_p=0.95)
                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # outputs = pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.3, top_k=10, top_p=1.0)
                    # output_text = outputs[0]["generated_text"]
                    output_text = output_text[output_text.rfind('[/INST]')+len('[/INST]'):]
                    if args.aug:
                        df.at[idx, "model_resp_aug"] = output_text
                    else:
                        df.at[idx, "model_resp"] = output_text
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

if __name__ == "__main__":
    main()