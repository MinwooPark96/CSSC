import torch
import pandas as pd
from datasets import DatasetDict, load_dataset, Dataset
from transformers import DataCollatorWithPadding

class AmazonDataset:
    def __init__(self, 
                 tokenizer, 
                 max_length = 128,
                 debug = False):
        
        
        self.id2label = {0: "negative", 1: "positive"}
        self.label2id = {"negative": 0, "positive": 1}
        
        self.train_raw_dataset = pd.read_csv('data/train_cleaned.csv') if not debug else pd.read_csv('data/train_cleaned.csv').head(1000)
        self.test_raw_dataset = pd.read_csv('data/test_cleaned.csv') if not debug else pd.read_csv('data/test_cleaned.csv').head(1000)
        
        self.dataset = DatasetDict({
            'train': Dataset.from_pandas(self.train_raw_dataset),
            'test': Dataset.from_pandas(self.test_raw_dataset)
        })
        
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.tokenized_dataset = self.dataset.map(self.tokenize_function,  batched=True)

    
    def tokenize_function(self, example):
        text = example["text"]
        
        tokenized_inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
        )
    
        return tokenized_inputs
