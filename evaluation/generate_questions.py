"""
Example Usage:

python evaluation/generate_questions.py --data_file <cultural_descriptions_file> --output_file <output_path> --pattern adapter --model mistralai/Mixtral-8x7B-Instruct-v0.1 --adapters <path_to_customized_adapter> --split full
"""

from transformers import AutoTokenizer
from tqdm import tqdm
from peft import PeftModel, AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import pandas as pd
import numpy as np
import os
import re
import json
import argparse
import transformers
import torch

from utils.prompt_utils import QUESTION_GENERATION_USER_TEMPLATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--pattern", type=str, choices=["merged", "adapter", "plain"])
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--adapters", default=[], nargs="+")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument(
        "--split", type=str, default="full", choices=["train", "test", "full"]
    )
    parser.add_argument("--sanity_check", action=argparse.BooleanOptionalAction)
    parser.add_argument("--partition", type=int, default=-1, choices=[-1, 0, 1, 2, 3])
    args = parser.parse_args()

    model_name = args.model
    tokenizer_path = args.tokenizer if args.tokenizer else model_name

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
                    bnb_4bit_use_double_quant=False,
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
                bnb_4bit_use_double_quant=False,
            ),
            attn_implementation="flash_attention_2",
        )
    else:
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    df = pd.read_csv(args.data_file)

    if args.split and args.split != "full":
        df = df.sample(frac=1, random_state=1234).reset_index(drop=True)
        train_split = 0.8
        if args.split == "train":
            df = df.head(int(len(df) * (train_split))).reset_index(drop=True)
        elif args.split == "test":
            test_split = (1.0 - train_split) / 2
            df = df.tail(int(len(df) * (test_split))).reset_index(drop=True)

    if args.partition != -1:
        partitions = np.array_split(df, 4)

        for i in range(len(partitions)):
            print(f"partition {i}:")
            print(partitions[i].head())
            print()
        df = partitions[args.partition]

        print(f"currently processing {len(df)} clusters")
        print(df.head())

    if args.num_samples != -1:
        df = df.sample(n=args.num_samples, replace=False, random_state=1234)
    if args.sanity_check:
        df = df.head(5)

    temperature = 0.7
    max_tokens = 1024
    top_p = 0.8
    top_k = 50
    # seed = 1234

    num_retries = 10
    num_samples = 1

    df_results = []
    for idx, _ in tqdm(df.iterrows(), total=len(df)):
        df_line = df.loc[idx]
        messages = []
        user_message = QUESTION_GENERATION_USER_TEMPLATE.format(df_line["desc"])
        for _ in range(num_samples):
            messages.append({"role": "user", "content": user_message})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            output_text = None
            for _ in range(num_retries):
                try:
                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                    outputs = text_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )

                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    output_text = output_text[
                        output_text.rfind("[/INST]") + len("[/INST]") :
                    ]

                    json_output = json.loads(output_text.strip())
                    if (
                        "Scenario" not in json_output
                        or "Persona" not in json_output
                        or "Question" not in json_output
                    ):
                        print(
                            f"returned json object is missing required fields: {json_output}"
                        )
                        continue

                    output_row = {}
                    output_row["cluster_id"] = df_line["cluster_id"]
                    output_row["scenario"] = json_output["Scenario"]
                    output_row["persona"] = json_output["Persona"]
                    output_row["question"] = json_output["Question"]

                    df_results.append(output_row)
                    messages.append({"role": "assistant", "content": output_text})
                    break
                except Exception as e:
                    print(f"encountered error at row {idx}: {e}")
                    if output_text:
                        print(f"model output: {output_text}")
                    print("retrying...")
                    continue

    df_results = pd.DataFrame.from_records(
        df_results, columns=["cluster_id", "scenario", "persona", "question"]
    )
    df_results.to_csv(args.output_file, index=None)


if __name__ == "__main__":
    main()
