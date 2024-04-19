import torch
import transformers
import argparse
import statistics
import math
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, AutoPeftModelForCausalLM, LoraConfig
from datasets import load_dataset

"""
Example Usage:

python evaluation/downstream_tasks/world_value_survey.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --pattern plain --batch_size 2
"""

def preprocess_data(example):
    examples = []
    selections = eval(example['selections'].replace("<class 'list'>", 'list'))
    options = eval(example['options'])
    for country in selections.keys():
      question_str = example['question']
      options_str = "\n".join([f"({chr(ord('A')+i)}) {x}" for i,x in enumerate(options)])
      input_text = f"Human: How would someone from {country} answer the following question:\n{question_str}\nHere are the options:\n{options_str}\nAssistant: If I had to select one of the options, my answer would be ("
      examples.append({'input_text': input_text, 'label': selections[country], 'country': country})
    return examples

def batch_data(examples, tokenizer):
    batched_data = []
    for i in range(0, len(examples), batch_size):
        batch_texts = [x['input_text'] for x in examples[i:i+batch_size]]
        batch_label = [x['label'] for x in examples[i:i+batch_size]]
        batch_country = [x['country'] for x in examples[i:i+batch_size]]
        batch_tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        batched_data.append({'input_text': batch_texts, 'input': batch_tokens, 'label': batch_label, 'country': batch_country})
    return batched_data

def JSDist(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    log_p = torch.log(p + eps)
    log_q = torch.log(q + eps)
    log_m = torch.log(m + eps)
    kl_p_m = F.kl_div(log_m, p, reduction='batchmean')
    kl_q_m = F.kl_div(log_m, q, reduction='batchmean')
    jsd = 0.5 * (kl_p_m + kl_q_m)
    
    # The square root of the Jensen–Shannon divergence is a metric often referred to as Jensen–Shannon distance
    return torch.sqrt(jsd).item()


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='model name')
    parser.add_argument("--pattern", type=str, default='plain', choices=["merged", "adapter", "plain", "awq"])
    parser.add_argument("--adapters", default=[], nargs='+')
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--resume_from', type=int, default=0, help='resume from which batch')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size to run the inference')
    parser.add_argument('--sanity_check', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    print("hello world")

    # Load dataset
    dataset = load_dataset("Anthropic/llm_global_opinions")
    dataset = dataset['train']

    # Load tokenizer and model
    model_name = args.model_name
    tokenizer_path = args.tokenizer if args.tokenizer else model_name

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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

    # hyper-parameters
    batch_size = args.batch_size

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = text_model

    # Preprocess the dataset
    examples = []
    for i, example in enumerate(dataset):
        examples.extend(preprocess_data(example))

    print(f'there are a total of {len(examples)} test samples')
    if args.sanity_check:
        examples = examples[:10]
        print(tokenizer.tokenize('(A)'))
        print(tokenizer.tokenize('A'))
        print(tokenizer.tokenize(' A'))

    batched_data = batch_data(examples, tokenizer)

    similarities = []
    sim_country = defaultdict(list)
    result = []

    print(f"batch_data_len:{len(batched_data)}")
    resume_from = args.resume_from

    # Loop through the dataset
    for batch_id, batch in tqdm(enumerate(batched_data[resume_from:]), total=len(batched_data[resume_from:])):

        inputs = batch['input']
        attention_mask = batch['input']['attention_mask']
        label = batch['label']

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        last_logits = logits[range(logits.size(0)), attention_mask.sum(dim=1) - 1, :]

        for i, cur_label in enumerate(label):
            tokens_to_check = [chr(ord('A')+j) for j in range(len(cur_label))]
            token_ids = tokenizer.convert_tokens_to_ids(tokens_to_check)
            prediction = F.softmax(last_logits[i, token_ids], dim=-1).to("cpu")

            
            # calculate sim = 1 - JSDist
            jsdist = JSDist(prediction, torch.tensor(cur_label))
            sim = 1.0 - jsdist
            if not math.isnan(sim):
                similarities.append(sim)
                sim_country[batch['country'][i]].append(sim)

    print("------------------------------------")
    if args.pattern == "adapter":
        model_name = model_name + "finetuned"
    print(f"benchmark results for {model_name}")
    print("overall average similarity:")
    print(sum(similarities)/len(similarities))

    # Calculate average loss for each country
    avg_sim_by_country = {}
    for country, sim in sim_country.items():
        if sim:
            avg_sim_by_country[country] = sum(sim) / len(sim)

    print("average sim by country:")
    print(avg_sim_by_country)
    print()

    sims = [v for k, v in avg_sim_by_country.items()]
    std = statistics.stdev(sims)
    print(f"standard deviation across countries: {std}")
    print()
    print()

if __name__ == "__main__":
    main()