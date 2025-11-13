import tokenize
from tokenizers import BertWordPieceTokenizer
import os 

CURRENT_DIR = os.path.dirname(__file__)
CORPUS_FILE_NAME = "chess_corpus.txt"
CORPUS_PATH = os.path.join(CURRENT_DIR, "..", CORPUS_FILE_NAME)

TOKENIZER_OUTPUT_DIR = os.path.join(CURRENT_DIR, "..", "chess_tokenizer")

special_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MOVESEP]", "[MASK]"
]

tokenizer = BertWordPieceTokenizer(
    clean_text = True,
    strip_accents = True,
    lowercase = False
)

print(f"starting to train tokenizer on {CORPUS_FILE_NAME}")

tokenizer.train(
    files = [CORPUS_PATH], 
    vocab_size = 1600, 
    min_frequency = 2, 
    special_tokens = special_tokens
)

if not os.path.exists(TOKENIZER_OUTPUT_DIR):
    os.makedirs(TOKENIZER_OUTPUT_DIR)

tokenizer.save_model(TOKENIZER_OUTPUT_DIR)

print(f"Training completed and the tokenizer output is saved in the {TOKENIZER_OUTPUT_DIR}")
print(f"Final Vocabulary size: {tokenizer.get_vocab_size()}")

