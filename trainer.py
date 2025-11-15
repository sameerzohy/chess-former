# --- Part 4: Define Arguments and Start Training ---
from transformers import TrainingArguments, Trainer
from configure_model import model, tokenizer, data_collector
import torch

from datasets import Dataset

import os

CURRENT_DIR = os.path.dirname(__file__)
TRAIN_FILE_NAME = "chess_train.txt"
TEST_FILE_NAME = "chess_test.txt"
TRAIN_PATH = os.path.join(CURRENT_DIR, "..", TRAIN_FILE_NAME)
TEST_PATH = os.path.join(CURRENT_DIR, "..", TEST_FILE_NAME)

def load_dataset(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    return Dataset.from_dict({"text": data})

train_dataset = load_dataset(TRAIN_PATH)
eval_dataset = load_dataset(TEST_PATH)

tokenized_datasets = {
    'train': train_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True),
    'validation': eval_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
}

OUTPUT_DIR_COLAB = "chess_bert_model"

# NOTE: These training arguments are conservative for a Colab T4 GPU
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR_COLAB,
    overwrite_output_dir=True,
    num_train_epochs=3, # Start with 3 epochs
    per_device_train_batch_size=4, # CRITICAL: Conservative size for Colab GPU
    per_device_eval_batch_size=4,
    save_steps=10000,
    save_total_limit=2, # Saves only the last two checkpoints
    prediction_loss_only=True,
    logging_steps=500,
    report_to="none",
    # Enable mixed precision (fp16) for faster training if Colab supports it
    fp16=torch.cuda.is_available() 
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collector,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

print("\nStarting BERT pre-training via Masked Language Modeling on Colab GPU...")

# This command starts the actual training process.
trainer.train() 

print("Training complete! Model and checkpoints saved in the 'chess_bert_model' folder.")