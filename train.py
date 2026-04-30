import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

# --- 設定 ---
MODEL_ID = "p1atypus/GPT-OSS-20B" # またはローカルパス
DATA_PATH = "train_data.jsonl"
OUTPUT_DIR = "./output-skeptical-core"

def train():
    # 1. トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. データセットのロード
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 3. モデルのロード (4-bit量子化 & マルチGPU分散)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", # 2枚のGPUに自動分配
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # 4060Ti (Ada) なので Flash Attention 2 を使用
        attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
    )

    # 4. LoRA設定
    # GPT-NeoX 系のターゲットモジュールは "query_key_value"
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. トレーニング引数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, # VRAM節約のため1
        gradient_accumulation_steps=16, # 実効バッチサイズ 16*2=32
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True, # bf16をサポートするGPU向け
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # 6. トレーナー
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 7. 学習開始
    print("Starting training...")
    trainer.train()

    # 8. 保存
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora"))
    print("Training finished and model saved.")

if __name__ == "__main__":
    train()
