from pathlib import Path
import os
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

training_files = [str(x) for x in Path("/content/drive/MyDrive/urdu_data").glob("**/*.txt")]
tokenizer = ByteLevelBPETokenizer()

for i, training_file in enumerate(tqdm(training_files, desc="Training")):
    with open(training_file, 'r', encoding='utf-8') as file:
        text = file.read()
        tokenizer.train(
            files=[training_file],
            vocab_size=30_522,
            min_frequency=2,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )

# Save the tokenizer
tokenizer.save("/content/drive/MyDrive/urdu_data/tokenizer.json")

# Save the model and config
tokenizer.save_model("/content/drive/MyDrive/urdu_data")