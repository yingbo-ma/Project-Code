import xlrd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

import warnings
warnings.filterwarnings(action='ignore')

import gensim

root_path = r"E:\Research Data\Data_Dialogue"
data_folder = r"\pair1\pair1_dialogue.xlsx"
file_path = root_path + data_folder

print("the data folder is in : ", file_path)

# read raw data from .xlsx file

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

# data_analysis = nltk.FreqDist(Speakers_List)
# data_analysis.plot(25, cumulative=False)
#
# data_analysis = nltk.FreqDist(Conversations_List)
# data_analysis.plot(25, cumulative=False)
#
# data_analysis = nltk.FreqDist(Tags_List)
# data_analysis.plot(25, cumulative=False)

# prepare sentence embedding

Text_Data = []

for sentence_index in range(len(Conversations_List)):
    utterance = Conversations_List[sentence_index]
    # words = tokenizer.tokenize(utterance) # remove punctuation
    words = nltk.word_tokenize(utterance) # keep punctuation
    print(words)

    temp_sentence = []
    for word in words:
        lower_case_word = word.lower()
        temp_sentence.append(lower_case_word) # keep original formats
        # stemmed_lower_case_word = ps.stem(lower_case_word)
        # temp_sentence.append(stemmed_lower_case_word) # use stem formats

    Text_Data.append(temp_sentence)

print(Text_Data)

# Create CBOW model

word2vec_model = gensim.models.Word2Vec(Text_Data, min_count = 1, size = 300, window = 5) #        size : Dimensionality of the word vectors.

word2vec_features = []

for sentence_index in range(len(Text_Data)):
    sentence = Text_Data[sentence_index]
    print(sentence)

    sentence_vector = []

    for word_index in range(len(sentence)):
        word = sentence[word_index]
        word_vector = word2vec_model.wv[word]
        sentence_vector.append(word_vector)
        sentence_vector = np.average(sentence_vector, axis=0)
        sentence_vector = sentence_vector.tolist()

    word2vec_features.append(sentence_vector)

word2vec_features = np.asarray(word2vec_features) # shape is (890 * 100)

# prepare tagging embedding

label_vectors = []

# for index in range(len(Tags_List)):
#
#     # for class 'ES'
#     if Tags_List[index] == 'ES' and Speakers_List[index] == 'David':
#         tag_vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
#     if Tags_List[index] == 'ES' and Speakers_List[index] == 'Luke':
#         tag_vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#     # for class 'EO'
#     if Tags_List[index] == 'EO' and Speakers_List[index] == 'David':
#         tag_vector = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
#     if Tags_List[index] == 'EO' and Speakers_List[index] == 'Luke':
#         tag_vector = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#     # for class 'D'
#     if Tags_List[index] == 'D' and Speakers_List[index] == 'David':
#         tag_vector = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
#     if Tags_List[index] == 'D' and Speakers_List[index] == 'Luke':
#         tag_vector = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#     # for class 'Q'
#     if Tags_List[index] == 'Q' and Speakers_List[index] == 'David':
#         tag_vector = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
#     if Tags_List[index] == 'Q' and Speakers_List[index] == 'Luke':
#         tag_vector = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
#     # for class 'U'
#     if Tags_List[index] == 'U' and Speakers_List[index] == 'David':
#         tag_vector = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
#     if Tags_List[index] == 'U' and Speakers_List[index] == 'Luke':
#         tag_vector = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
#     # for class 'P'
#     if Tags_List[index] == 'P' and Speakers_List[index] == 'David':
#         tag_vector = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
#     if Tags_List[index] == 'P' and Speakers_List[index] == 'Luke':
#         tag_vector = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
#     # for class 'A'
#     if Tags_List[index] == 'A' and Speakers_List[index] == 'David':
#         tag_vector = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
#     if Tags_List[index] == 'A' and Speakers_List[index] == 'Luke':
#         tag_vector = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
#     # for class 'OR'
#     if Tags_List[index] == 'OR' and Speakers_List[index] == 'David':
#         tag_vector = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
#     if Tags_List[index] == 'OR' and Speakers_List[index] == 'Luke':
#         tag_vector = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
#     # for class 'OU'
#     if Tags_List[index] == 'OU' and Speakers_List[index] == 'David':
#         tag_vector = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
#     if Tags_List[index] == 'OU' and Speakers_List[index] == 'Luke':
#         tag_vector = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
#     # for class 'NA'
#     if Tags_List[index] == 'NA' and Speakers_List[index] == 'David':
#         tag_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
#     if Tags_List[index] == 'NA' and Speakers_List[index] == 'Luke':
#         tag_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]

for index in range(len(Tags_List)):

    # for class 'ES'
    if Tags_List[index] == 'ES':
        tag_vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # for class 'EO'
    if Tags_List[index] == 'EO':
        tag_vector = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # for class 'D'
    if Tags_List[index] == 'D':
        tag_vector = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # for class 'Q'
    if Tags_List[index] == 'Q':
        tag_vector = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # for class 'U'
    if Tags_List[index] == 'U':
        tag_vector = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # for class 'P'
    if Tags_List[index] == 'P':
        tag_vector = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    # for class 'A'
    if Tags_List[index] == 'A':
        tag_vector = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # for class 'OR'
    if Tags_List[index] == 'OR':
        tag_vector = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    # for class 'OU'
    if Tags_List[index] == 'OU':
        tag_vector = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    # for class 'NA'
    if Tags_List[index] == 'NA':
        tag_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    # for class 'IN'
    if Tags_List[index] == 'IN':
        tag_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


    label_vectors.append(tag_vector)

label_vectors = np.asarray(label_vectors) # shape is (890 * 12)

# prepare sequence data

num_timesteps = 5 # time step of lstm
sequence_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(word2vec_features)-num_timesteps+1)] # get the index for each utterance senquence

sequence_data = []
sequence_target = []

for i in sequence_index:
    temp = []

    temp.append(word2vec_features[i[0][0]])
    temp.append(word2vec_features[i[1][0]])
    temp.append(word2vec_features[i[2][0]])
    temp.append(word2vec_features[i[3][0]])
    temp.append(word2vec_features[i[4][0]])

    sequence_data.append(temp)

for i in sequence_index:
    temp = []

    temp.append(label_vectors[i[0][0]])
    temp.append(label_vectors[i[1][0]])
    temp.append(label_vectors[i[2][0]])
    temp.append(label_vectors[i[3][0]])
    temp.append(label_vectors[i[4][0]])

    sequence_target.append(temp)

sequence_data = np.asarray(sequence_data) # (890, 5, [number of words in that sentence])
sequence_target = np.asarray(sequence_target) # (890, 5, 12)

print(sequence_data.shape)
print(sequence_target.shape)

# define sequential model

from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, TimeDistributed, LSTM, Dense, Input, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.optimizers import Adam
from sklearn.metrics import classification_report

n_classes = 11
train_test_split = 0.7
train_num = int(len(sequence_data) * train_test_split)

sequential_model = Sequential()
sequential_model.add(Bidirectional(LSTM((64), return_sequences=True)))
sequential_model.add(Bidirectional(LSTM((64), return_sequences=True)))
sequential_model.add(Dense(n_classes, activation="softmax", kernel_regularizer=regularizers.l2(0.01)))
sequential_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

train_X = sequence_data[0: train_num]
test_X = sequence_data[train_num : ]

train_Y = sequence_target[0: train_num]
test_Y = sequence_target[train_num : ]

# start training

sequential_model.fit(
    train_X,
    train_Y,
    batch_size = 12,
    epochs = 100
)

sequential_model.summary()

# start testing

_, test_acc = sequential_model.evaluate(
    test_X,
    test_Y
)

print('Accuracy: %.2f' % (test_acc * 100))
