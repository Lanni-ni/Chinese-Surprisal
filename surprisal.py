import os
import torch
import openpyxl
import numpy as np
import pandas as pd
from minicons import scorer
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertForMaskedLM, BertTokenizer


MODEL_NAME = {"gpt2-chn": "uer/gpt2-chinese-cluecorpussmall"}
FEATURE_IN_LINEAR = ['Surprisal', 'i', 'l', 'f', 'ao', 'o']

def calc_surprisal(words):
    sentences = ["".join(words)]
    model = scorer.IncrementalLMScorer('user/gpt2-chinese-cluecorpussmall','cpu')
    model.compute_stats(model.prepare_text(sentences))
    output = model.token_score(sentences, surprisal = True, base_two = True)

    character_surprisal = output[0]

    word_surprisal = []
    for word in words:
        surprisal_sum = 0
        for char in word:
            for char_surprisal in character_surprisal:
                if char_surprisal[0] == char:
                    surprisal_sum += char_surprisal[1]
                    break
        word_surprisal.append((word, surprisal_sum))

    return word_surprisal
def sentences_prediction(grouped_words):
    surprisal_list = []
    for words in grouped_words:
        words_surprisal = calc_surprisal(words)
        surprisal_value = [row[1] for row in words_surprisal]
        surprisal_list += surprisal_value
    df['Surprisal'] = surprisal_list
    df.to_excel("sentences_prediction.xlsx", index = False)