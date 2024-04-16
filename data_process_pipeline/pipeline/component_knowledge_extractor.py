from transformers import AutoTokenizer
from tqdm import tqdm
from peft import PeftModel, AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModel, file_utils
import transformers

transformers.logging.set_verbosity_info()
cache_dir = file_utils.default_cache_path
print(f"Default cache directory: {cache_dir}")
import os
import transformers
import torch
import random
import json
import argparse
import logging
import pandas as pd
import numpy as np

from utils.util import parse_to_int, process_output
from utils.prompt_utils import get_mixtral_user_prompt, reencode_prompt_utf16, truncate_to_token_limit, KNOWLEDGE_EXTRACTION_FIELDS

logger = logging.getLogger(__name__)


class ClusteringComponent(PipelineComponent):
    description = "extracting structured cultural knowledge from social media comments"
    config_layer = "knowledge_extractor"
    
    def __init__(self, config: dict):
        super().__init__(config)

        # get local config
        self._local_config = config[self.config_layer]        
        self.sanity_check = self._local_config['sanity_check']
        
        
        model_name = self._local_config["model"]
        tokenizer_path = self._local_config["tokenizer"] if "tokenizer" in self._local_config else model_name
        if self._local_config["pattern"] == "adapter":
            adapters = self._local_config["adapters"]
            assert len(adapters) >= 1
            text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if len(adapters) == 1:
                # No need to merge
                text_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    # load_in_4bit=True,
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
                    bnb_4bit_use_double_quant=False,
                ),
                attn_implementation="flash_attention_2",
            )
        else:
            raise NotImplementedError
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        df = pd.read_csv(self._local_config["data_file"])
        
        if self._local_config["num_samples"] != -1:
            df = df.sample(n=self._local_config["num_samples"], replace=False, random_state=1234)
        elif self._local_config["partition"] != -1:
            assert self._local_config["partition"] < self._local_config["num_partitions"]
            partitions = np.array_split(df, self._local_config["num_partitions"])

            for i in range(len(partitions)):
                logger.info(f"partition {i}:")
                logger.info(partitions[i].head())
                logger.info('\n')
            df = partitions[self._local_config["partition"]]

            logger.info(f"currently processing {len(df)} clusters")
            logger.info(df.head())


        if self.sanity_check:
            df = df.head(5)
        
        self.tokenizer = tokenizer
        self.df = df
        self.text_model = text_model
    
    def run(self):
        self.df["has_culture"] = False
        self.df["model_resp"] = ""
        self.df["json_output"] = ""
        
        text_model = self.text_model
        tokenizer = self.tokenizer

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                df_line = self.df.loc[idx]
                if self.sanity_check:
                    logger.info(df_line["submission_title"])
                    logger.info(df_line["comment_content"])
                    logger.info()

                submission = (
                    ""
                    if pd.isna(df_line["submission_title"])
                    else str(df_line["submission_title"])
                )
                comment = (
                    ""
                    if pd.isna(df_line["comment_content"])
                    else str(df_line["comment_content"])
                )
                comment = truncate_to_token_limit(comment, max_tokens=12000)

                # zero shot inference without in-context examples
                user_message = get_mixtral_user_prompt(submission, comment)

                if len(user_message) * 4 > 24000:
                    logger.warning("user message too long, continuing...")
                    continue

                # zero shot inference without in-context examples
                messages = [{"role": "user", "content": user_message}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt = reencode_prompt_utf16(prompt)
                if self.sanity_check:
                    logger.info(prompt)
                    logger.info('\n')

                num_retries = 10
                output_text = None
                for _ in range(num_retries):
                    try:
                        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                        outputs = text_model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.1,
                            top_k=10,
                            top_p=0.2,
                        )
                        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                        # outputs = pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.3, top_k=10, top_p=1.0)
                        # output_text = outputs[0]["generated_text"]
                        output_text = output_text[
                            output_text.rfind("[/INST]") + len("[/INST]") :
                        ]
                        self.df.at[idx, "model_resp"] = output_text

                        if self.sanity_check:
                            logger.info("model's output:")
                            logger.info(output_text)

                        has_culture, outputs = process_output(output_text)
                        if not has_culture:
                            break

                        self.df.at[idx, "has_culture"] = True
                        for output in outputs:
                            for field in CULTUREBANK_FIELDS:
                                assert field in output
                        self.df.at[idx, "json_output"] = json.dumps(outputs)

                        break
                    except Exception as e:
                        logger.error(e)
                        if output_text:
                            logger.error("generated output:")
                            logger.error(output_text)
                        logger.error(f"error generating output at line {idx}, retrying...")
            except Exception as e:
                logger.error(e)
                logger.error(f"error encountered at line {idx}, continuing...")
                continue
    
    def save_output(self):
        self.df.to_csv(self._local_config["output_file"], index=None)