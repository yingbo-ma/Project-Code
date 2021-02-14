import xlrd
import numpy as np
import matplotlib.pyplot as plt

import re

from support_functions import decontracted

import nltk
from nltk.tokenize import sent_tokenize

import warnings
warnings.filterwarnings(action='ignore')

corpus_path = "C:\\Users\\Yingbo\\Desktop\\ENGAGE dataset\\Corpus.xlsx"

# read raw data from .xlsx file
Raw_TimeStep_List = []
Raw_Speakers_List = []
Raw_Text_List = []

book = xlrd.open_workbook(corpus_path)
sheet = book.sheet_by_index(0)

print("the number of rows is :", sheet.nrows)
print("the number of coloumns is :", sheet.ncols)

for row_index in range(0, sheet.nrows): # skip heading and 1st row
    time, speaker, text = sheet.row_values(row_index, end_colx=4)

    if speaker == 'Other':
        print('row index of', row_index, 'is removed becaused of Other Speaker.')
    elif text == '(())':
        print('row index of ', row_index, 'is removed becaused of Null Text.')
    elif type(text) == int or type(text) == float:
        text = str(text)
        Raw_TimeStep_List.append(time)
        Raw_Speakers_List.append(speaker)
        Raw_Text_List.append(text)
    else:
        Raw_TimeStep_List.append(time)
        Raw_Speakers_List.append(speaker)
        Raw_Text_List.append(text)

TimeStep_Corpus = []
Speakers_Corpus = []
Text_Corpus = []

print(len(Raw_TimeStep_List))
print(len(Raw_Speakers_List))
print(len(Raw_Text_List))

# start preprocessing
for text_index in range(len(Raw_Text_List)):

    time_step = Raw_TimeStep_List[text_index]
    speaker = Raw_Speakers_List[text_index]
    utterance = Raw_Text_List[text_index]

    # step 1: replace all digital numbers with consistent string of 'number'
    utterance = re.sub(r'\d+', 'number', utterance)
    # step 2: convert text to lowercase
    utterance = utterance.lower()
    # step 3: solve contractions
    utterance = decontracted(utterance)
    # step 4: remove '()' several times. '((())) happens in text corpus'
    utterance = re.sub(r'\([^()]*\)', '', utterance)
    utterance = re.sub(r'\([^()]*\)', '', utterance)
    utterance = re.sub(r'\([^()]*\)', '', utterance)
    utterance = re.sub(r'\([^()]*\)', '', utterance)
    utterance = re.sub(r'\([^()]*\)', '', utterance)
    # step 5: remove '[]' and contents inside it
    utterance = re.sub("([\[]).*?([\]])", "", utterance)
    # step 6: remove specific symbols, keep ? and . because it conveys the question and answer info
    utterance = re.sub(r'[--]', '', utterance) # '--'
    utterance = re.sub(r'[(]', '', utterance)  # '('
    # utterance = re.sub(r'[...]', '', utterance)  # '...' this would remove all . remove it later
    utterance = re.sub(r'[\\]', '', utterance)  # '\'
    utterance = re.sub(r'[\']', '', utterance)  # '''
    utterance = re.sub(r'[!]', '', utterance)  # '!'
    utterance = re.sub(r'["]', '', utterance)  # '"'
    utterance = re.sub(r'[{]', '', utterance)  # '{"}'
    utterance = re.sub(r'[}]', '', utterance)  # '}'
    # step 7: remove white space
    utterance = utterance.strip()
    # step 6: sentence tokenize
    utterance = sent_tokenize(utterance)

    for sentence_index in range(len(utterance)):
        TimeStep_Corpus.append(time_step)
        Speakers_Corpus.append(speaker)
        sub_utterance = utterance[sentence_index]
        sub_utterance = re.sub(r'[...]', '', sub_utterance) # remove '...'
        if sub_utterance == '':
            sub_utterance = 'inaudible'
        Text_Corpus.append(sub_utterance)

print(len(TimeStep_Corpus))
print(len(Speakers_Corpus))
print(len(Text_Corpus))

print(TimeStep_Corpus)
print(Speakers_Corpus)
print(Text_Corpus)

data_analysis = nltk.FreqDist(TimeStep_Corpus)
data_analysis.plot(25, cumulative=False)

data_analysis = nltk.FreqDist(Speakers_Corpus)
data_analysis.plot(25, cumulative=False)

data_analysis = nltk.FreqDist(Text_Corpus)
data_analysis.plot(25, cumulative=False)

np.save("timestep.npy", TimeStep_Corpus)
np.save("speaker.npy", Speakers_Corpus)
np.save("text.npy", Text_Corpus)