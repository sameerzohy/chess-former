from sklearn.model_selection import train_test_split 
import os 

CURRENT_DIR = os.path.dirname(__file__)
CORPUS_FILE_NAME = "chess_corpus.txt"
CORPUS_PATH = os.path.join(CURRENT_DIR, "..", CORPUS_FILE_NAME)

TRAIN_FILE_NAME = "chess_train.txt"
TEST_FILE_NAME = "chess_test.txt"

TRAIN_PATH = os.path.join(CURRENT_DIR, "..", TRAIN_FILE_NAME)
TEST_PATH = os.path.join(CURRENT_DIR, "..", TEST_FILE_NAME)

try: 
    with open(CORPUS_PATH, 'r') as f:
        sequences = f.read().splitlines()

    print(f" Loaded count: {len(sequences)} sequences.")

    train_sequence, test_sequence = train_test_split(
        sequences,
        test_size = 0.20, 
        random_state = 42
    )

    with open(TRAIN_PATH, 'w') as f: 
        f.write('\n'.join(train_sequence))
    
    with open(TEST_PATH, 'w') as f: 
        f.write('\n'.join(test_sequence))
    
    print(f"Training sequences saved: {len(train_sequence)} to {TRAIN_FILE_NAME}")
    print(f"Testing sequences saved: {len(test_sequence)} to {TEST_FILE_NAME}")
    
except FileNotFoundError:
    print(f"Error: Corpus file not found at {CORPUS_PATH}. You may need to run Step 1.2 again.")
    
