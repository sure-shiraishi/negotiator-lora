from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os

# --- Config ---
MODEL_ID = "unsloth/gpt-oss-20b"
MAX_SEQ_LENGTH = 2048
DATA_PATH = "train_data.jsonl"
OUTPUT_DIR = "./output-skeptical-core-0511a"


def train():

    print(f"Loading model: {MODEL_ID}...")
    max_memory = {0: "14GB", 1: "14GB"}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        full_finetunign=False,
        max_memory=max_memory,
        device_map="auto",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.01,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print("Trainable parameters configured.")

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"Dataset loaded with {len(dataset)} examples.")

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=40,
        learning_rate=5e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    print("Starting training with Unsloth...")
    trainer.train()

    save_dir = os.path.join(OUTPUT_DIR, "skeptical_core_lora")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Training finished. LoRA saved to {save_dir}")


if __name__ == "__main__":
    train()
