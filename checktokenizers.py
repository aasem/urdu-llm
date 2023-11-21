from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
example = "Here is a dog"

# Tokenize the text
tokens = old_tokenizer.tokenize(example)

print(tokens)