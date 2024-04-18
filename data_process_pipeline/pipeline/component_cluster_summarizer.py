from transformers import AutoTokenizer
from tqdm import tqdm
from peft import PeftModel, AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import os
import ast
import transformers
import torch
import random
import json
import logging
import pandas as pd
import numpy as np

from utils.constants import SUMMARIZER_FIELDS
from utils.prompt_utils import SUMMARIZER_USER_PROMPT_TEMPLATE, SUMMARIZER_INCONTEXT_FIELDS_EXAMPLES, truncate_to_token_limit
from utils.util import parse_to_int
from pipeline.pipeline_component import PipelineComponent


logger = logging.getLogger(__name__)


class ClusterSummarizer(PipelineComponent):
    description = "summarize clustered cultural indicators"
    config_layer = "cluster_summarizer"
    
    def __init__(self, config: dict):
        super().__init__(config)

        # get local config
        self._local_config = config[self.config_layer]        
        self.sanity_check = self._local_config['sanity_check']

        model_name = self._local_config["model"]

        if self._local_config["pattern"] == "adapter":
            adapters = self._local_config["adapters"]
            assert len(adapters) >= 1
            text_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
            if len(adapters) == 1:
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
            elif len(adapters) > 1:
                # Need to merge
                text_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            logger.info("----------------------------------------------------")
            logger.info(f"Loaded the model {model_name}")

            if len(adapters) == 1:
                text_model = PeftModel.from_pretrained(text_model, adapters[0])
                
                logger.info("--------------------NO MERGING----------------------")
                logger.info(f"Loaded the adapter model {adapters[0]}")
            elif len(adapters) > 1:
                for adapter_name in adapters:
                    text_model = PeftModel.from_pretrained(text_model, adapter_name)
                    text_model = text_model.merge_and_unload()
            
                    logger.info("----------------------MERGING-----------------------")
                    logger.info(f"Loaded the adapter model {adapter_name}")
        elif self._local_config["pattern"] == "merged" or self._local_config["pattern"] == "plain":
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
        else:
            raise NotImplementedError

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")


        df = pd.read_csv(self._local_config["input_file"])
        random.seed(123)

        if self._local_config["filter_threshold"] != -1:
            df = df.loc[df["cluster_size"] >= self._local_config["filter_threshold"]]
            logger.info(f"a total of {len(df)} clusters with size >= {self._local_config['filter_threshold']}")
        if self._local_config["num_samples"] != -1:
            df = df.sample(n=self._local_config["num_samples"], replace=False, random_state=12)
        if self._local_config["partition"] != -1:
            assert self._local_config["partition"] < self._local_config["num_partitions"]
            partitions = np.array_split(df, elf._local_config["num_partitions"])

            for i in range(len(partitions)):
                logger.info(f"partition {i}:")
                logger.info(partitions[i].head())
                logger.info("\n")
            df = partitions[self._local_config["partition"]]

            logger.info(f"currently processing {len(df)} clusters")
            logger.info(df.head())

        if self.sanity_check:
            df = df.head(10)
        self.df = df
        self.text_model = text_model
        self.tokenizer = tokenizer
    
    def run(self):
        df = self.df
        text_model = self.text_model
        tokenizer = self.tokenizer
        
        df_before_cluster = pd.read_csv(self._local_config["original_before_cluster_file"])
        dict_before_cluster = df_before_cluster.set_index(
            df_before_cluster["vid_unique"]
        ).T.to_dict()

        df_results = []
        max_samples = 10
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                df_line = df.loc[idx]
                raw_samples = df_line["raw_samples"]
                raw_samples = []
                for unique_id in eval(df_line["raw_sample_vids"]):
                    raw_sample = {
                        field: dict_before_cluster[unique_id][field]
                        for field in SUMMARIZER_FIELDS + ["norm"]
                    }
                    raw_samples.append(raw_sample)
                
                
                norms = [raw_sample for raw_sample in raw_samples if parse_to_int(raw_sample["norm"]) is not None and parse_to_int(raw_sample["norm"]) == 1]
                not_norms = [raw_sample for raw_sample in raw_samples if parse_to_int(raw_sample["norm"]) is not None and parse_to_int(raw_sample["norm"]) == 0]
                
                raw_samples = norms if len(norms) >= len(not_norms) else not_norms
                
                cluster_samples = []
                for raw_sample in raw_samples:
                    cluster_sample = {
                        field: raw_sample[field]
                        for field in SUMMARIZER_FIELDS
                    }
                    cluster_samples.append(cluster_sample)

                if len(cluster_samples) > max_samples:
                    cluster_samples = random.sample(cluster_samples, max_samples)

                user_prompt = (
                    SUMMARIZER_USER_PROMPT_TEMPLATE.format(
                        json.dumps(cluster_samples), json.dumps(SUMMARIZER_INCONTEXT_FIELDS_EXAMPLES)
                    )
                )
                                
                # zero shot inference without in-context examples
                messages = [{"role": "user", "content": user_prompt}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt = truncate_to_token_limit(prompt)
                
                num_retries = 10
                
                for _ in range(num_retries):
                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                    outputs = text_model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.3, top_k=10, top_p=1.0)
                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    output_text = output_text[output_text.rfind('[/INST]')+len('[/INST]'):]
                    try:
                        start_index, end_index = output_text.find('{'), output_text.rfind('}')
                        json_string = output_text[start_index:end_index+1]
                        res = json.loads(json_string)
                        valid = True
                        expected_fields = SUMMARIZER_FIELDS + ['topic']
                        for key in res:
                            if key not in expected_fields:
                                logger.error(f"{df_line['cluster_id']}: output contains invalid field(s), retrying...")
                                valid = False
                                break
                        if valid:
                            res["cluster_id"] = df_line['cluster_id']
                            df_results.append(res)
                            break
                    except Exception as e:
                        logger.error(e)
                        logger.error("generated output:")
                        logger.error(output_text)
                        logger.error(f"error generating output at cluster {df_line['cluster_id']}, retrying...")
            except Exception as e:
                logger.error(e)
                logger.error(f"error encountered at cluster {idx}, continuing...")
                continue
        self.save_output(df_results)
        logger.info("Cluster Summarization Done!")
    
    def save_output(self, df_results):
        df_results = pd.DataFrame.from_records(df_results, columns=["cluster_id"] + SUMMARIZER_FIELDS + ['topic'])
        df_results.to_csv(self._local_config["output_file"], index=None)

