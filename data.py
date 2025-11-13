# Cooking here.......


# imoport statements
import pandas as pd
import os 

# kitchen - cooking 

CURRENT_DIR = os.path.dirname(__file__)
FILE_PATH = os.path.join(CURRENT_DIR, "..", "positions.csv")
FEN_COLUMN = "fen"
MOVE_COLUMN = "playing"

try: 
    df = pd.read_csv(
        FILE_PATH,
        usecols=[FEN_COLUMN, MOVE_COLUMN], 
        low_memory=True, # the large files are processed in chunks.
    )

    df.rename(columns={FEN_COLUMN: "FEN", MOVE_COLUMN: "Move"}, inplace=True)

    #ouput log: 

    # print("\nSample of the loaded data:")
    # print(df.head())

    # print("\nLast 5 rows of the loaded data:")
    # print(df.tail())

except FileNotFoundError: 
    print(f"File not found at {FILE_PATH}, Please check the file path and try again.")
    df = None 

# step 2 : Format the sequence: 

SEPERATOR_TOKEN = " [MOVESEP] " 

df['Sequence'] = df['FEN'] + SEPERATOR_TOKEN + df['Move']

OUTPUT_FILE_NAME = "chess_corpus.txt"
OUTPUT_PATH = os.path.join(CURRENT_DIR, "..", OUTPUT_FILE_NAME)

df['Sequence'].to_csv(OUTPUT_PATH, index=False, header=False)

print('Created the output file.')

