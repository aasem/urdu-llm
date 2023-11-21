from tokenizers import ByteLevelBPETokenizer

# Specify the path to your saved vocab and merges files
vocab = "21112023/vocab.json"
merges = "21112023/merges.txt"

# Load the tokenizer
tokenizer = ByteLevelBPETokenizer(vocab, merges)

# Prepare some Urdu text to tokenize.
# Ideally, this should be a large corpus for performance testing.
text = "ایک ہوں مسلم حرم کی پاسبانی کے لیے۔ نیل کے ساحل سے لے کر تابخاک کاشغر"

# Tokenize the text
tokens = tokenizer.encode(text)

# Output results
print(f"Tokenized {len(text)} characters.")
print(f"Number of tokens: {len(tokens.tokens)}")
print(f"The Decoded Tokens: {tokenizer.decode(tokens.ids)}")
# Print the encoded tokens
print("Tokens:", tokens.tokens)

# If you want to see the offsets which tell you where each token is found in the original text
print("Offsets:", tokens.offsets)

# Print each token with its corresponding piece of original text
for token, (start, end) in zip(tokens.tokens, tokens.offsets):
    print(f"Token: {token}, Original Text: '{text[start:end]}'")