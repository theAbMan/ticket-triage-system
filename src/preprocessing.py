import re
from datasets import load_dataset

dataset = load_dataset("trec")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]","",text)
    text = re.sub(r"\s+"," ",text).strip()
    return text


sample = dataset['train'][0]['text']
print("Before: ", sample)
print("After: ", clean_text(sample))