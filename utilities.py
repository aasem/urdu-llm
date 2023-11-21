import re
import os
from tqdm.auto import tqdm

def preprocess(text):
    """
    This function takes a string of text as input and performs several cleaning steps:
    - Replaces standard commas with Urdu commas.
    - Normalizes whitespace, ensuring that punctuation marks are followed by a space.
    - Reduces sequences of punctuation to a single instance.
    - Removes hyperlinks.
    - Removes English words.
    - Removes strings of dashes and periods.
    - Removes long strings of the same number.
    - Removes unwanted characters while preserving Urdu characters, numbers, common punctuation, and whitespace.
    - Removes disjointed numbers.
    """
    # Replace standard commas with Urdu commas
    text = text.replace(',', '،')

    # Normalize whitespace and ensure punctuation is followed by a space.
    text = re.sub(r'(?<=[۔؟!،:;])(?![\s])', r' ', text)

    # Remove hyperlinks
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove English words
    text = re.sub(r'\b[a-zA-Z]+\b', '', text)

    # Remove series of disjointed numbers and potential accompanying punctuation
    text = re.sub(r'\b\d+([،.]\s?\d+)*\b', '', text)

    # Remove long strings of the same number
    text = re.sub(r'(\b\d)\1+\b', '', text)

    # Remove sequences of repeated numbers
    text = re.sub(r'(\b\d+\s*){2,}', '', text)

    # Remove strings of repeated punctuation (، or . or ۔) including sequences like "۔ ۔ ۔"
    text = re.sub(r'([،.۔])\1+', r'\1', text)

    # Remove sequences of punctuation characters (with or without spaces)
    text = re.sub(r'[،.؟!؛:; ]+', ' ', text)  # Remove sequences of common punctuation (with or without spaces)

    # Remove sequences of mixed punctuation like ".،.. -.،."
    text = re.sub(r'[،.۔-]+', ' ', text)

    # Normalize whitespace again after the replacements
    text = re.sub(r'\s+', ' ', text)

    # Remove unwanted characters while preserving Urdu text, digits, and common punctuation
    text = re.sub(r'[^\u0600-\u06FF0-9,;:.!?\'\"\s\u0610-\u061A\u064B-\u065F\u0660-\u0669\u06D6-\u06ED\-]', '', text)

    # Remove newline characters
    text = text.replace('\n', ' ')

    # Strip leading and trailing whitespace
    return text.strip()

def preprocess_and_write_files(dataset, save_dir, file_size_limit=10_000_000):  # 10MB limit
    file_count = 0
    current_size = 0
    text_data = []

    for sample in tqdm(dataset, desc="Processing samples"):
        processed_sample = preprocess(sample['text'])
        # Estimate the size after encoding to UTF-8
        sample_size = len(processed_sample.encode('utf-8'))
        if current_size + sample_size > file_size_limit:
            # Write the current batch to file
            with open(os.path.join(save_dir, f'ur_{file_count}.txt'), 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            # Reset for the next batch
            text_data = [processed_sample]
            current_size = sample_size
            file_count += 1
        else:
            text_data.append(processed_sample)
            current_size += sample_size

    # Write any remaining samples to a file
    if text_data:
        with open(os.path.join(save_dir, f'ur_{file_count}.txt'), 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))