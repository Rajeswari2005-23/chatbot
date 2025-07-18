import argparse
import logging
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_function(examples, tokenizer):
    prompts = [
        f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
        for instr, inp in zip(examples["instruction"], examples["input"])
    ]
    full_texts = [prompt + out for prompt, out in zip(prompts, examples["output"])]
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=1024,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_local_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Ensure consistent output format (all strings)
    for item in data:
        if not isinstance(item["output"], str):
            item["output"] = json.dumps(item["output"], ensure_ascii=False)

    assert all(k in data[0] for k in ["instruction", "input", "output"]), "Missing keys"
    return Dataset.from_list(data)


def main(args):
    logger.info("Loading dataset...")
    dataset = load_local_dataset(args.data_path)
    split = dataset.train_test_split(test_size=0.2)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Preparing 4-bit quantized model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    model = get_peft_model(model, lora_config)

    logger.info("Tokenizing dataset...")
    tokenized = split.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=split["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        num_train_epochs=5,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=50,
        warmup_steps=100,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        optim="adamw_torch",
        max_grad_norm=0.3,
        weight_decay=0.01,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
    )

    model.print_trainable_parameters()

    logger.info("Training...")
    trainer.train()

    logger.info("Saving model and tokenizer...")
    model.save_pretrained(args.save_model_dir)
    tokenizer.save_pretrained(args.save_model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="teknium/OpenHermes-2.5-Mistral-7B")
    parser.add_argument("--data_path", type=str, default="test.json")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--save_model_dir", type=str, default="./outputs/final_model")
    args = parser.parse_args()
    main(args)
