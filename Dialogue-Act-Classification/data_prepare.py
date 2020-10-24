import xlrd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()

import warnings
warnings.filterwarnings(action='ignore')

import gensim

root_path = r"E:\Research Data\Data_Dialogue"
data_folder = r"\pair1\pair1_dialogue.xlsx"
file_path = root_path + data_folder

print("the data folder is in : ", file_path)

Speakers_List = []
Conversations_List = []
Tags_List = []


book = xlrd.open_workbook(file_path)
sheet = book.sheet_by_index(0)

print("the number of rows is :", sheet.nrows)
print("the number of coloumns is :", sheet.ncols)

for row_index in range(2, sheet.nrows): # skip heading and 1st row
    speaker, time, utterance, code, notes = sheet.row_values(row_index, end_colx=5)

    Speakers_List.append(speaker)
    Conversations_List.append(utterance)
    Tags_List.append((code))

print(Speakers_List)
print(Conversations_List)
print(Tags_List)

data_analysis = nltk.FreqDist(Speakers_List)
data_analysis.plot(25, cumulative=False)

data_analysis = nltk.FreqDist(Conversations_List)
data_analysis.plot(25, cumulative=False)

data_analysis = nltk.FreqDist(Tags_List)
data_analysis.plot(25, cumulative=False)

# prepare sentence embedding
Text_Data = []

for sentence_index in range(len(Conversations_List)):
    utterance = Conversations_List[sentence_index]
    words = nltk.word_tokenize(utterance)

    temp_sentence = []
    for word in words:
        lower_case_word = word.lower()
        stemmed_lower_case_word = ps.stem(lower_case_word)
        temp_sentence.append(stemmed_lower_case_word)

    Text_Data.append(temp_sentence)

print(Text_Data)

# Create CBOW model
model = gensim.models.Word2Vec(Text_Data, min_count = 1, size = 100, window = 5) #        size : Dimensionality of the word vectors.

word2vec_features = []

for sentence_index in range(len(Text_Data)):
    sentence = Text_Data[sentence_index]
    print(sentence)

    sentence_vector = []

    for word_index in range(len(sentence)):
        word = sentence[word_index]
        word_vector = model.wv[word]
        sentence_vector.append(word_vector)

    word2vec_features.append(sentence_vector)

word2vec_features = np.asarray(word2vec_features, dtype=object)

# prepare tagging embedding

label_vectors = []

for index in range(len(Tags_List)):
    if Tags_List[index] == 'ES' and Speakers_List[index] == 'David':
        tag_vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif Tags_List[index] == 'ES' and Speakers_List[index] == 'Luke':
        tag_vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    else:
        print("error")


