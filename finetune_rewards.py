import os
import sys
from typing import List
import numpy as np
import fire
import torch
import torch.nn as nn
import evaluate
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    TaskType,
    set_peft_model_state_dict,
)
from transformers import Trainer, DataCollatorWithPadding
from transformers import LlamaForSequenceClassification, LlamaForCausalLM, LlamaTokenizer

from trl import RewardTrainer

from utils.prompter import Prompter

class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForSequenceClassification.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        num_labels=1
    )

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, config)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "right"  # Allow batched inference


    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt_chosen = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["chosen"],
        )
        tokenized_full_prompt_chosen = tokenize(full_prompt_chosen)
        full_dict = dict()
        full_dict["input_ids_chosen"] = tokenized_full_prompt_chosen["input_ids"]
        full_dict["attention_mask_chosen"] = tokenized_full_prompt_chosen["attention_mask"]

        full_prompt_rejected = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["rejected"],
        )
        tokenized_full_prompt_rejected = tokenize(full_prompt_rejected)
        full_dict["input_ids_rejected"] = tokenized_full_prompt_rejected["input_ids"]
        full_dict["attention_mask_rejected"] = tokenized_full_prompt_rejected["attention_mask"]

        return full_dict

    train_args = transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                logging_steps=10,
                optim="adamw_torch",
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=200 if val_set_size > 0 else None,
                save_steps=200,
                output_dir=output_dir,
                save_total_limit=5,
                bf16=True,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
            )
    
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    
    data = load_dataset("json", data_files=data_path)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    )

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True


    # Define the metric that we'll use for validation.
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        # Here, predictions is rewards_j and rewards_k.
        # We want to see how much of the time rewards_j > rewards_k.
        predictions = np.argmax(predictions, axis=0)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[SavePeftModelCallback],
        args=train_args,
        compute_metrics=compute_metrics
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
