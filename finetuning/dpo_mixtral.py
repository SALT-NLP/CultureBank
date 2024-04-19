# Adapted from: https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py

# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig, deepspeed

from trl import DPOTrainer

"""
Example Usage:

accelerate launch finetuning/dpo_mixtral.py --model_name_or_path=<your_model_name> --model_type='plain' --output_dir=<output_directory> --max_length 4096 --data_file=<path_to_training_data> --run_name <experiment_name> --warmup_steps 50 --gradient_accumulation_steps 8 --num_train_epochs 2 --report_to='wandb'
"""

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    tokenizer: Optional[str] = field(default=None, metadata={"help": "path to tokenizer"})
    model_type: Optional[str] = field(
        default="plain",
        metadata={"help": "the type of the model: plain, merge, or peft"},
    )
    adapter: Optional[str] = field(default="", metadata={"help": "path to peft adaptor"})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=4096, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=8192, metadata={"help": "the maximum sequence length"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "max number of training steps"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    run_name: Optional[str] = field(default="dpo_mixtral", metadata={"help": "the name of the run"})
    data_file: Optional[str] = field(default="/nlp/scr/zyanzhe/Maple/matplotlib_qa_all_preference_v0_6K.csv", metadata={"help": "data_file"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def get_stack_exchange_paired(
    data_file,
    data_dir,
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }
    """
    if data_dir == "training":
        # dataset = load_dataset("csv", data_files=data_file, split="train[:95%]")
        dataset = load_dataset("csv", data_files=data_file, split="train")
    elif data_dir == "evaluation":
        dataset = load_dataset("csv", data_files=data_file, split="train[95%:]")
    else:
        dataset = None

    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": samples["query"],
            "chosen": samples["good"],
            "rejected": samples["bad"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


def train():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK') or 0)}") if ddp else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tokenizer_path = script_args.tokenizer if script_args.tokenizer else script_args.model_name_or_path
    
    if script_args.model_type == "plain":
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            device_map=device_map,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2"
        )
    elif script_args.model_type == "merge":
        model = AutoPeftModelForCausalLM.from_pretrained(script_args.adapter, device_map="cpu", torch_dtype=torch.float16)
        model = model.merge_and_unload()
        # model = model.to(device)
        
        merged_checkpoint = os.path.join(script_args.adapter, "final_merged_checkpoint")
        model.save_pretrained(merged_checkpoint)
        
        model = AutoModelForCausalLM.from_pretrained(
            merged_checkpoint,
            device_map=device_map,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2"
        )
    elif script_args.model_type == "peft":
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            device_map=device_map,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2"
        )
        model = PeftModel.from_pretrained(model, script_args.adapter)
    else:
        raise NotImplementedError
    
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # model ref can be another model.
    """
    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    """

    model_ref = None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Stack-exchange paired dataset
    train_dataset = get_stack_exchange_paired(data_file=script_args.data_file, data_dir="training", sanity_check=script_args.sanity_check)
    train_dataset = train_dataset.filter(
        lambda x: len(tokenizer(x["prompt"] + x["chosen"]).input_ids) <= script_args.max_length
        and len(tokenizer(x["prompt"] + x["chosen"]).input_ids) <= script_args.max_length
    )

    # 3. Load evaluation dataset
    eval_dataset = get_stack_exchange_paired(data_file=script_args.data_file, data_dir="evaluation", sanity_check=True)
    eval_dataset = eval_dataset.filter(
        lambda x: len(tokenizer(x["prompt"] + x["chosen"]).input_ids) <= script_args.max_length
        and len(tokenizer(x["prompt"] + x["chosen"]).input_ids) <= script_args.max_length
    )

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        # if not -1, will override epochs
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        tf32=True,
        remove_unused_columns=False,
        run_name=script_args.run_name,
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    if script_args.model_type == "peft":
        peft_config = None

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)


if __name__ == "__main__":
    train()