import xlrd
import numpy as np
import matplotlib.pyplot as plt

import re
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize

ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

utterance = ' Oh, yeah. Write it down, then, because you already had the other [crosstalk 00:01:01]-'

# step 1: remove numbers
print('before step 1: ', utterance)
utterance = re.sub(r'\d+', '', utterance)
print('after step 1: ', utterance)
# step 2: convert text to lowercase
utterance = utterance.lower()
print('after step 2: ', utterance)
# step 3: solve contractions
utterance = decontracted(utterance)
print('after step 3: ', utterance)
# step 4: remove '()' several times. '((())) happens in text corpus'
utterance = re.sub(r'\([^()]*\)', '', utterance)
utterance = re.sub(r'\([^()]*\)', '', utterance)
utterance = re.sub(r'\([^()]*\)', '', utterance)
utterance = re.sub(r'\([^()]*\)', '', utterance)
utterance = re.sub(r'\([^()]*\)', '', utterance)
print('after step 4: ', utterance)
# step 5: remove '[]' and contents inside it
utterance = re.sub("([\[]).*?([\]])", "", utterance)
print('after step 5: ', utterance)
# step 6: remove specific symbols, keep ? and . because it conveys the question and answer info
utterance = re.sub(r'[--]', '', utterance) # '--'
utterance = re.sub(r'[(]', '', utterance)  # '('
# utterance = re.sub(r'[...]', '', utterance)  # '...'
utterance = re.sub(r'[\\]', '', utterance)  # '\'
utterance = re.sub(r'[\']', '', utterance)  # '''
utterance = re.sub(r'[!]', '', utterance)  # '!'
utterance = re.sub(r'["]', '', utterance)  # '"'
print('after step 6: ', utterance)
# step 7: remove white space
utterance = utterance.strip()
print('after step 7: ', utterance)
# step 8: sentence tokenize
utterance = sent_tokenize(utterance)
print('after step 8: ', utterance)