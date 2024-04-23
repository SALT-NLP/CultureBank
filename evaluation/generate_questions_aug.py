"""
Example Usage:

python evaluation/generate_questions_aug.py --data_file <cultural_descriptions_file_or_preliminary_questions> --output_file <output_path> --knowledge_file <cultural_descriptions_file> --pattern adapter --model mistralai/Mixtral-8x7B-Instruct-v0.1 --adapters <path_to_customized_adapter> --split full
"""

from transformers import AutoTokenizer
from tqdm import tqdm
from peft import PeftModel, AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI

import pandas as pd
import numpy as np
import os
import re
import json
import math
import argparse
import transformers
import torch
import openai

from utils.prompt_utils import (
    QUESTION_GENERATION_USER_TEMPLATE,
    QUESTION_EVAL_SYSTEM_PROMPT,
    QUESTION_EVAL_USER_TEMPLATE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--knowledge_file", type=str)
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

    eval_questions = [
        "Is the generated question relevant to the given knowledge?",
        "Does the generated question **indirectly** refer to the given knowledge?",
    ]
    eval_thresholds = [0.95, 0.8]

    model_name = args.model
    tokenizer_path = args.tokenizer if args.tokenizer else model_name
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()

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
                bnb_4bit_use_double_quant=False,
            ),
            attn_implementation="flash_attention_2",
        )
    else:
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    df = pd.read_csv(args.data_file)
    knowledge = pd.read_csv(args.knowledge_file)

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
        df = df.head(10)

    temperature = 0.7
    max_tokens = 1024
    top_p = 0.8
    top_k = 50
    # seed = 1234

    gpt_model = "gpt-4-1106-preview"
    gpt_temperature = 0
    gpt_max_tokens = 1
    gpt_top_p = 0.01
    gpt_seed = 1234

    num_retries = 10
    num_samples = 1

    df_results = []
    for idx, _ in tqdm(df.iterrows(), total=len(df)):
        df_line = df.loc[idx]
        knowledge_line = knowledge.loc[
            knowledge["cluster_id"] == df_line["cluster_id"]
        ].iloc[0]
        messages = []
        user_message = QUESTION_GENERATION_USER_TEMPLATE.format(knowledge_line["desc"])
        best_score = 0
        for _ in range(num_samples):
            messages.append({"role": "user", "content": user_message})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            json_output = {
                "Scenario": df_line["scenario"] if "scenario" in df.columns else "",
                "Persona": df_line["persona"] if "persona" in df.columns else "",
                "Question": df_line["question"] if "question" in df.columns else "",
            }
            output_text = json.dumps(json_output, indent=4)
            output_row = {}
            output_row["cluster_id"] = df_line["cluster_id"]
            for _ in range(num_retries):
                try:
                    good_question = True
                    curr_score = 0
                    for i in range(len(eval_questions)):
                        eval_question = eval_questions[i]
                        eval_threshold = eval_thresholds[i]
                        eval_messages = [
                            {"role": "system", "content": QUESTION_EVAL_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": QUESTION_EVAL_USER_TEMPLATE.format(
                                    knowledge_line["desc"],
                                    json_output["Question"],
                                    eval_question,
                                ),
                            },
                        ]
                        response = client.chat.completions.create(
                            model=gpt_model,
                            messages=eval_messages,
                            seed=gpt_seed,
                            temperature=gpt_temperature,
                            max_tokens=gpt_max_tokens,
                            top_p=gpt_top_p,
                            logprobs=True,
                            top_logprobs=5,
                        )

                        top_logprobs = (
                            response.choices[0].logprobs.content[0].top_logprobs
                        )

                        yes_prob = 0
                        for logprob in top_logprobs:
                            if logprob.token == "Yes":
                                yes_prob = math.exp(logprob.logprob)
                                curr_score += yes_prob
                                break

                        if args.sanity_check:
                            print(top_logprobs)
                            print(f"yes_prob: {yes_prob}")

                        if yes_prob < eval_threshold:
                            good_question = False

                    if good_question:
                        output_row["scenario"] = json_output["Scenario"]
                        output_row["persona"] = json_output["Persona"]
                        output_row["question"] = json_output["Question"]
                        break

                    if curr_score > best_score:
                        best_score = curr_score
                        output_row["scenario"] = json_output["Scenario"]
                        output_row["persona"] = json_output["Persona"]
                        output_row["question"] = json_output["Question"]

                    print(
                        f"The generated question does not meet the quality standard, retrying..."
                    )
                    print(f"Knowledge: {knowledge_line['desc']}")
                    print(f"Generated question: {json_output['Question']}")
                    print()
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

                except Exception as e:
                    print(f"encountered error at row {idx}: {e}")
                    if output_text:
                        print(f"model output: {output_text}")
                    print("retrying...")
                    continue
            df_results.append(output_row)
            messages.append({"role": "assistant", "content": output_text})

    df_results = pd.DataFrame.from_records(
        df_results, columns=["cluster_id", "scenario", "persona", "question"]
    )
    df_results.to_csv(args.output_file, index=None)


if __name__ == "__main__":
    main()
