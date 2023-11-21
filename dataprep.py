# login with -> huggingface-cli login
# add token to access datasets

from datasets import load_dataset_builder, load_dataset
from tqdm.auto import tqdm
from utilities import preprocess, preprocess_and_write_files
import os

ds_builder = load_dataset_builder("oscar-corpus/OSCAR-2301", "ur")
dataset = load_dataset("oscar-corpus/OSCAR-2301", "ur")
oscar_ds = dataset['train']

sample = preprocess(oscar_ds[0]['text'])
print(sample)

# Set the directory where you want to save the files
save_dir = '/content/drive/My Drive/urdu_data/'
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

preprocess_and_write_files(oscar_ds, save_dir)