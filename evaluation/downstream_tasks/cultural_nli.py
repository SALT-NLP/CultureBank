import collections
import numpy as np
import re
import scipy
import torch
import argparse
import transformers
import random
import json

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, AutoPeftModelForCausalLM, LoraConfig
from torch.nn.functional import softmax


"""
Example Usage:

python evaluation/downstream_tasks/cultural_nli.py --pattern plain --model mistralai/Mixtral-8x7B-Instruct-v0.1 --output_file <output_json_file> --batch_size 1
"""


def process_nli(premise: str, hypothesis: str, gt_group: str):
  """ process to required xnli format with task prefix """
  user_message = NLI_TEMPLATE.format(premise, hypothesis, CULTURE_MAP[gt_group])

  return user_message

def get_majority_label(label_str):
  label_to_name = {'C': 'contradict', 'N': 'neutral', 'E': 'entail'}
  labels = eval(label_str)
  majority_label = collections.Counter(labels).most_common(1)[0][0]
  return label_to_name[majority_label]

def run_inference_batched(premise_hypothesis_pair_batch, model, gt_group, batch_size=4):
  seqs = [process_nli(premise=premise, hypothesis=hypothesis, gt_group=gt_group) for
      premise, hypothesis in premise_hypothesis_pair_batch]
  print(len(seqs))
  print(seqs[0])
  nli_scores = collections.defaultdict(dict)
  for b_i in range(0, len(seqs), batch_size):
      premises, hypotheses = zip(*premise_hypothesis_pair_batch[b_i: b_i + batch_size])
      seq_batch = seqs[b_i : b_i + batch_size]
      inputs = tokenizer(seq_batch, padding=True, truncation=True, return_tensors="pt").to(device)

      outputs = model(**inputs)
      attention_mask = inputs['attention_mask']
      logits = outputs.logits
      last_logits = logits[range(logits.size(0)), attention_mask.sum(dim=1) - 1, :]
      scores = last_logits[:, label_inds]
      probs = softmax(scores, dim=1)
      label_id_batch = torch.argmax(probs, dim=1).tolist()
      id_to_label = {0: "entail", 1: "neutral", 2: "contradict"}
      for i, label_id in enumerate(label_id_batch):
          prediction = id_to_label[label_id]
          nli_scores[premises[i]][hypotheses[i]] = {'label': prediction}
          nli_scores[premises[i]][hypotheses[i]].update(
              dict(zip([id_to_label[j] for j in range(3)], probs[i].tolist())))
  return nli_scores

def run_inference_batched_roberta(premise_hypothesis_pairs, batch_size=4):
  nli_scores = collections.defaultdict(dict)
  for b_i in range(0, len(premise_hypothesis_pairs), batch_size):
    premises, hypotheses = zip(*premise_hypothesis_pairs[b_i: b_i + batch_size])
    input_batch = tokenizer(list(premises), list(hypotheses),
                            return_tensors='pt', max_length=512, padding=True, truncation=True)
    for k in input_batch:
      input_batch[k] = input_batch[k].to(device)
    with torch.no_grad():
      logits = model(**input_batch).logits
    probs = logits.softmax(dim=1)
    label_id_batch = torch.argmax(probs, dim=1).tolist()
    for i, label_id in enumerate(label_id_batch):
      prediction = model.config.id2label[label_id]
      nli_scores[premises[i]][hypotheses[i]] = {'label': prediction}
      nli_scores[premises[i]][hypotheses[i]].update(
          dict(zip([model.config.id2label[i] for i in range(3)], probs[i].tolist())))
    print('Finished %d' % (b_i + batch_size))
  return nli_scores

def get_majority_vote_if_exist(xs):
  m = scipy.stats.mode(xs, keepdims=False)[0]
  return m if xs.count(m) > (len(xs) / 2) else None


def get_combined_label(g_to_rs):
  l = get_majority_vote_if_exist([r for rs in g_to_rs.values() for r in rs])
  if l == 0:
    l = 'contradict'
  elif l == 2:
    l = 'entail'
  else:
    l = 'neutral'
  return l


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--pattern", type=str, choices=["merged", "adapter", "plain", "awq"])
  parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
  parser.add_argument("--tokenizer", type=str, default=None)
  parser.add_argument("--adapters", default=[], nargs='+')
  parser.add_argument('--batch_size', type=int, default=4, help='batch size to run the inference')
  parser.add_argument("--output_file", type=str)
  parser.add_argument("--sanity_check", action=argparse.BooleanOptionalAction)
  args = parser.parse_args()


  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


  model_name = args.model
  tokenizer_path = args.tokenizer if args.tokenizer else model_name
  batch_size = args.batch_size

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
  tokenizer.pad_token = tokenizer.eos_token


  raw_data = np.loadtxt('/sailhome/lansong/CultureBank/evaluation/downstream_tasks/data.tsv', delimiter='\t', skiprows=1, dtype=str)
  cali_data = collections.defaultdict(dict)

  ENTAILS_LABEL = tokenizer.tokenize("(E)")[1]
  NEUTRAL_LABEL = tokenizer.tokenize("(N)")[1]
  CONTRADICTS_LABEL = tokenizer.tokenize("(C)")[1]

  label_inds = tokenizer.convert_tokens_to_ids(
      [ENTAILS_LABEL, NEUTRAL_LABEL, CONTRADICTS_LABEL]
  )


  if args.sanity_check:
    entail_tokens = tokenizer.tokenize("(E)")
    neutral_tokens = tokenizer.tokenize("(N)")
    contradict_tokens = tokenizer.tokenize("(C)")

    print("Entail tokens:", entail_tokens)
    print("Neutral tokens:", neutral_tokens)
    print("Contradict tokens:", contradict_tokens)
    print(f"Label ids: {label_inds}")



  NLI_TEMPLATE = """Premise: {}
  Hypothesis: {}

  Let’s think as someone who lives in {}. What do you think is the relationship between the premise and the hypothesis?
  (E) Entail 
  (N) Neutral
  (C) Contradict

  Your Answer (E/N/C): ("""

  CULTURE_MAP = {
      "US": "the United States",
      "IN": "India",
  }

  for row in raw_data:
    cali_data[str(row[0])][str(row[1])] = {'labels': {'US': get_majority_label(str(row[2])), 'IN': get_majority_label(str(row[3]))}}
  NLI_CLASSIFICATION_DATA = dict(cali_data)


  # full_class_name_mapping = {'entail': 'entailment', 'contradict': 'contradiction', 'neutral': 'neutral'}
  full_class_name_mapping = {'entail': 'entail', 'contradict': 'contradict', 'neutral': 'neutral'}
  eval_class = 'entail'

  test_set = NLI_CLASSIFICATION_DATA
  ordered_pairs = [(p, h) for p, hs in test_set.items() for h in hs]

  if args.sanity_check:
      ordered_pairs = ordered_pairs[:10]

  results = {}


  print(f"benchmark results for {args.model}")
  for gt_group in ('US', 'IN'): 
      predictions = run_inference_batched(ordered_pairs, text_model, gt_group, batch_size=batch_size)
      if args.sanity_check:
          print(predictions)
      results[gt_group] = predictions

      ax = None
      f1_scores = []

      labels = [int(test_set[p][h]['labels'][gt_group] == eval_class) for p, h in ordered_pairs]
      preds = [predictions[p][h][full_class_name_mapping[eval_class]]  for p, h in ordered_pairs]
      print(len(labels), sum(labels))

      f1 = max([f1_score(labels, [int(x > t) for x in preds], average='macro') for t in (0.6, 0.7, 0.8, 0.9, 1.0)])
      f1_scores.append(f1)
      print(f'f1 score for {gt_group}: {f1}')
  # print(' & '.join(map(lambda x: '%.2f' % x, f1_scores)))
  print()

  with open(args.output_file, 'w') as f:
    json.dump(results, f)
    
if __name__ == "__main__":
    main()