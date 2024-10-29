import random
import torch
import numpy as np
import pandas as pd
import re
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer


def set_seed(seed: int = 42):
    # Set seed for python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy random number generator
    np.random.seed(seed)
    
    # Set seed for PyTorch on the CPU
    torch.manual_seed(seed)
    
    # Set seed for PyTorch on the GPU (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you have multi-GPU setup
    
    # Make sure that CUDA operations are deterministic (may slow down performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def concat_columns(df: pd.DataFrame, col1: str, col2: str, new_col: str) -> pd.DataFrame:
    df[new_col] = df[col1].apply(str) + ' ' + df[col2].apply(str)
    df.drop(col2, axis = 1, inplace = True)
    return df

def clean_text(text: str) -> str:
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-zÀ-ú ]+', '', text)
    # Analyzing the most used words below, i chose to exclude these because there are too many and are unnecessary
    text = re.sub('book|one', '', text)
    # Convert to lower case
    text = text.lower()
    # remove scores
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(texto):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(texto.lower())
    return " ".join([token for token in tokens if token not in stop_words])

def normalize_text(text):
    stemmer = SnowballStemmer("english")
    normalized_text = []
    for word in text.split():
        stemmed_word = stemmer.stem(word)
        normalized_text.append(stemmed_word)
    return ' '.join(normalized_text)