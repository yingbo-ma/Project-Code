import xlrd
import numpy as np
import matplotlib.pyplot as plt

import re

from support_functions import decontracted

import warnings
warnings.filterwarnings(action='ignore')

corpus_path = r"E:\Research Data\ENGAGE\ENGAGE Recordings\Dec4-2019 - t007 t091\clean_data\Dec4-2019 - t007 t091_cleaned_by_Kiana.xlsx"

# read raw data from .xlsx file
Raw_Text_List = []
Clean_Text_List = []

tag_list = []

book = xlrd.open_workbook(corpus_path)
sheet = book.sheet_by_index(0)

for row_index in range(1, sheet.nrows): # skip heading and 1st row
    time, speaker, text, tag = sheet.row_values(row_index, end_colx=4)

    if speaker == 'Other':
        print('row index of', row_index, 'is removed becaused of Other Speaker.')
    elif text == '(())':
        print('row index of ', row_index, 'is removed becaused of Null Text.')
    elif type(text) == int or type(text) == float:
        text = str(text)
        Raw_Text_List.append(text)
    else:
        Raw_Text_List.append(text)
        if ('Impasse' in tag):
            tag_list.append(1)
        else:
            tag_list.append(0)

# start preprocessing
for text_index in range(len(Raw_Text_List)):

    utterance = Raw_Text_List[text_index]

    # step 1: replace all digital numbers with consistent string of 'number'
    utterance = re.sub(r'\d+', 'number', utterance)
    # step 2: convert text to lowercase
    utterance = utterance.lower()
    # step 3: solve contractions
    utterance = decontracted(utterance)
    # step 4: remove '()' several times. '((())) happens in text corpus'
    utterance = re.sub('[()]', '', utterance)
    # step 5: remove '[]' and contents inside it
    utterance = re.sub("([\[]).*?([\]])", "", utterance)
    # step 6: remove specific symbols, keep ? and . because it conveys the question and answer info
    utterance = re.sub(r'[--]', '', utterance) # '--'
    utterance = re.sub(r'[\\]', '', utterance)  # '\'
    utterance = re.sub(r'[\']', '', utterance)  # '''
    utterance = re.sub(r'[!]', '', utterance)  # '!'
    utterance = re.sub(r'["]', '', utterance)  # '"'
    utterance = re.sub(r'[{]', '', utterance)  # '{"}'
    utterance = re.sub(r'[}]', '', utterance)  # '}'
    # step 7: remove '...'
    if '...' in utterance:
        utterance = utterance.replace('...', "")
    # step 8: remove white space
    utterance = utterance.strip()
    Clean_Text_List.append(utterance)

# data_analysis = nltk.FreqDist(Text_Corpus)
# data_analysis.plot(25, cumulative=False)

# for index in range(len(Clean_Text_List)):
#     print(str(index+2) + ": " + Clean_Text_List[index])

# prepare the adjacent utterance pair
utterance_pair = []

for index in range(len(Clean_Text_List) - 1):
    temp_utterance = Clean_Text_List[index] + " " + Clean_Text_List[index + 1]
    utterance_pair.append(temp_utterance)

# print out final results
for index in range(len(utterance_pair)):
    print("Pair " + str(index) + " is: " + utterance_pair[index] + ". The corresponding impasse Tag is: " + str(tag_list[index]))

tag_list = tag_list[:len(utterance_pair)]
print(tag_list.count(1))

combined_data = np.column_stack((utterance_pair, tag_list))
print(combined_data)