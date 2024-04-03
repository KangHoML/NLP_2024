from datasets import load_dataset
import os

data_dir = './data'
dataset = load_dataset('nsmc')
os.makedirs(data_dir, exist_ok=True)
for split_key in dataset.keys():
    doc_path = f"{data_dir}/{split_key}.txt"
    with open(doc_path, 'w') as f:
        for doc in dataset[split_key]['document']:
            f.write(doc+'\n')
