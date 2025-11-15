from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from tokenizers import BertWordPieceTokenizer
from transformers import PreTrainedTokenizerFast 
import os
import torch 

CURRENT_DIR = os.path.dirname(__file__)
TOKENIZER_OUTPUT_DIR = os.path.join(CURRENT_DIR, "..", "chess_tokenizer")
TRAIN_FILE_NAME = "chess_train.txt"
TEST_FILE_NAME = "chess_test.txt"
TRAIN_PATH = os.path.join(CURRENT_DIR, "..", TRAIN_FILE_NAME)
TEST_PATH = os.path.join(CURRENT_DIR, "..", TEST_FILE_NAME)


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file = os.path.join(TOKENIZER_OUTPUT_DIR, "tokenizer.json"), 
    unk_token = "[UNK]", 
    sep_token = "[SEP]", 
    pad_token = "[PAD]", 
    cls_token = "[CLS]", 
    mask_token = "[MASK]", 
    additional_special_token = ["[MOVESEP]"]
)

config = BertConfig(
    vocab_size = tokenizer.vocab_size, 
    hidden_size = 768, 
    num_hidden_layers = 12, 
    num_attention_heads = 12, 
    intermediate_size = 3072, 
    max_position_embeddings = 512,
    model_max_length = 512,
    #tokenizer id's
    pad_token_id = tokenizer.pad_token_id, 
    mask_token_id = tokenizer.mask_token_id, 
    sep_token_id = tokenizer.sep_token_id, 
    cls_token_id = tokenizer.cls_token_id, 
)

model = BertForMaskedLM(config=config)
print(f"Model initialized with vocabulary size: {config.vocab_size}")
print(f"Total model Parameters: {model.num_parameters}")


data_collector = DataCollatorForLanguageModeling(
    tokenizer = tokenizer, 
    mlm = True, 
    mlm_probability = 0.15
)