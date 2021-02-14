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
# data_folder = r"\pair1\pair1_dialogue.xlsx"
data_folder = r"\Corpus.xlsx"
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

    if code == 'NA':
        print('row index of ', row_index, 'is removed becaused of NA.')
    elif code == 'IN':
        print('row index of ', row_index, 'is removed because of IN.')
    elif code == '':
        print('row index of ', row_index, 'is removed because of EMPTY.')
    else:
        Speakers_List.append(speaker)
        Conversations_List.append(utterance)
        Tags_List.append(code)

# prepare sentence data
Text_Data = []

for sentence_index in range(len(Conversations_List)):
    utterance = Conversations_List[sentence_index]
    words = tokenizer.tokenize(utterance) # remove punctuation

    temp_sentence = []
    for word in words:
        lower_case_word = word.lower()
        temp_sentence.append(lower_case_word)  # keep original formats

    Text_Data.append(temp_sentence)

print('the length of text data is ', len(Text_Data))

# Create CBOW model
word2vec_model = gensim.models.Word2Vec(Text_Data, min_count = 1, size = 300, window = 2) # size : Dimensionality of the word vectors.

word2vec_features = []

for sentence_index in range(len(Text_Data)):
    sentence = Text_Data[sentence_index]
    sentence_vector = []

    for word_index in range(len(sentence)):
        word = sentence[word_index]
        word_vector = word2vec_model.wv[word]
        sentence_vector.append(word_vector)

    word2vec_features.append(sentence_vector)

word2vec_features = np.asarray(word2vec_features)
print("the shape of sentence embedding is ", word2vec_features.shape)

# pad the input sentence encoding
from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX_LEN = 10

padded_sentence_encoding = pad_sequences(word2vec_features, padding="post", truncating="post", maxlen=MAX_LEN)
print("padded sentence shape is ", padded_sentence_encoding.shape)

# prepare DA tagging embedding

da_label_vectors = []

for index in range(len(Tags_List)):

    tag_vector = []

    # for class 'ES'
    if Tags_List[index] == 'ES':
        tag_vector = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    # for class 'EO'
    if Tags_List[index] == 'EO':
        tag_vector = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    # for class 'D'
    if Tags_List[index] == 'D':
        tag_vector = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    # for class 'Q'
    if Tags_List[index] == 'Q':
        tag_vector = [0, 0, 0, 1, 0, 0, 0, 0, 0]
    # for class 'U'
    if Tags_List[index] == 'U':
        tag_vector = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    # for class 'P'
    if Tags_List[index] == 'P':
        tag_vector = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    # for class 'A'
    if Tags_List[index] == 'A':
        tag_vector = [0, 0, 0, 0, 0, 0, 1, 0, 0]
    # for class 'OR'
    if Tags_List[index] == 'OR':
        tag_vector = [0, 0, 0, 0, 0, 0, 0, 1, 0]
    # for class 'OU'
    if Tags_List[index] == 'OU':
        tag_vector = [0, 0, 0, 0, 0, 0, 0, 0, 1]

    da_label_vectors.append(tag_vector)

da_label_vectors = np.asarray(da_label_vectors)

# prepare speaker-turn taggingg embedding

st_label_vectors = [] # st: speaker-turn

print(Speakers_List)

for index in range(len(Speakers_List)):
    if index == len(Speakers_List) - 1:
        st_label_vectors.append(0)
    elif Speakers_List[index] == Speakers_List[index + 1]:
        st_label_vectors.append(0)
    else:
        st_label_vectors.append(1)

print(st_label_vectors)
print(len(st_label_vectors))


# prepare sequence data

num_timesteps = 3 # time step of lstm
sequence_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(padded_sentence_encoding)-num_timesteps+1)] # get the index for each utterance senquence

sequence_data = []
sequence_da_target = []
sequence_st_target = []

for i in sequence_index:
    temp = []

    temp.append(padded_sentence_encoding[i[0][0]])
    temp.append(padded_sentence_encoding[i[1][0]])
    temp.append(padded_sentence_encoding[i[2][0]])
    # temp.append(word2vec_features[i[3][0]])
    # temp.append(word2vec_features[i[4][0]])

    sequence_data.append(temp)

for i in sequence_index:
    temp = []

    temp.append(da_label_vectors[i[0][0]])
    temp.append(da_label_vectors[i[1][0]])
    temp.append(da_label_vectors[i[2][0]])
    # temp.append(label_vectors[i[3][0]])
    # temp.append(label_vectors[i[4][0]])

    sequence_da_target.append(temp)

for i in sequence_index:
    temp = []

    temp.append(st_label_vectors[i[0][0]])
    temp.append(st_label_vectors[i[1][0]])
    temp.append(st_label_vectors[i[2][0]])
    # temp.append(label_vectors[i[3][0]])
    # temp.append(label_vectors[i[4][0]])

    sequence_st_target.append(temp)

sequence_data = np.asarray(sequence_data) # (890, 5, [number of words in that sentence])
sequence_da_target = np.asarray(sequence_da_target) # (890, 5, 9)
sequence_st_target = np.asarray(sequence_st_target)

print("sequence_data", sequence_data.shape)
print("sequence_da_target", sequence_da_target.shape)
print("sequence_st_target", sequence_st_target.shape)

print(sequence_st_target)
print(st_label_vectors)

# building keras model

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Bidirectional
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop
from keras_contrib.layers import CRF
from sklearn.metrics import classification_report
from tensorflow.keras.utils import plot_model

# define utterance embedding layer
In_SEN_DIM = sequence_data.shape[3]
OUT_SEN_DIM = 128
print("input sentence dim is ", MAX_LEN, "*", In_SEN_DIM)
print("output sentence dim is ", OUT_SEN_DIM)

# speaker turn classification model
inputs = Input(
    shape=(num_timesteps, MAX_LEN, In_SEN_DIM),
    name='main_input'
)

utter_embedding = LSTM(OUT_SEN_DIM, return_sequences=False)
main_branch = TimeDistributed(utter_embedding)(inputs)
main_branch = Bidirectional(LSTM(128, return_sequences=True, activation="relu"))(main_branch)
main_branch = Bidirectional(LSTM(64, return_sequences=True, activation="relu"))(main_branch)

da_output = Dense(9, activation="softmax", kernel_regularizer=regularizers.l2(0.01), name="da_output")(main_branch)
st_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), name="st_output")(main_branch)

model = Model(
    inputs = inputs,
    outputs = [da_output, st_output]
)

model.compile(
    optimizer=RMSprop(lr=0.2, decay=0.001),
    loss={"da_output":"categorical_crossentropy", "st_output":"binary_crossentropy"},
    loss_weights={"da_output":0.5, "st_output":1.0},
    metrics=['accuracy']
)

model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.fit(
    {"main_input":sequence_data},
    {"da_output":sequence_da_target, "st_output":sequence_st_target},
    batch_size = 32,
    epochs = 20
)