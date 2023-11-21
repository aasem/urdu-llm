import chardet

with open("/content/drive/MyDrive/urdu_data/ur_1.txt", 'rb') as file:
    result = chardet.detect(file.read())
print(result['encoding'])