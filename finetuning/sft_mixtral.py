"""
Example Usage:

accelerate launch finetuning/sft_mixtral.py --model_name_or_path=<your_model_name> --output_dir=<output_directory> --max_seq_length 4096 --dataset_name=<name_or_path_to_input_dataset> --run_name <experiment_name> --warmup_steps 500 --gradient_accumulation_steps 16 --num_train_epochs 8
"""

# Adapted from: https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Dict, Optional
from accelerate import Accelerator

import torch
import os
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from trl import (
    ModelConfig,
    SFTTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.import_utils import is_npu_available, is_xpu_available


tqdm.pandas()


@dataclass
class ScriptArguments:
    dataset_name: str = field(
        metadata={"help": "the dataset name"},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "the text field of the dataset"}
    )
    max_seq_length: int = field(
        default=8192, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )
    tokenizer: str = field(default=None, metadata={"help": "tokenizer path"})


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################

    # manually initializing the model and training configs
    model_config = ModelConfig(
        model_name_or_path=model_config.model_name_or_path,
        torch_dtype="bfloat16",
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        use_bnb_nested_quant=True,
        attn_implementation="flash_attention_2",
        use_peft=True,
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )

    training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=-1,
        logging_steps=10,
        save_steps=100,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs,
        group_by_length=False,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=training_args.warmup_steps,
        weight_decay=0.05,
        optim="paged_adamw_32bit",
        bf16=True,
        remove_unused_columns=False,
        run_name=training_args.run_name,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=True,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
        quantization_config=quantization_config,
    )

    print(model_kwargs)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        **model_kwargs,
    )

    base_model.config.use_cache = False
    peft_config = get_peft_config(model_config)
    print(peft_config)

    tokenizer_path = (
        args.tokenizer if args.tokenizer else model_config.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    ################
    # Dataset
    ################
    train_dataset = load_dataset("csv", data_files=args.dataset_name)["train"]
    print(train_dataset)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field=args.dataset_text_field,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        peft_config=peft_config,
        dataset_batch_size=256,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
