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
        print('row index of ', row_index, 'is removed becaused of NA tagging.')
    elif code == 'IN':
        print('row index of ', row_index, 'is removed because of IN tagging.')
    elif code == '':
        print('row index of ', row_index, 'is removed because of EMPTY tagging.')
    else:
        Speakers_List.append(speaker)
        Conversations_List.append(utterance)
        Tags_List.append(code)

data_analysis = nltk.FreqDist(Speakers_List)
data_analysis.plot(25, cumulative=False)

data_analysis = nltk.FreqDist(Conversations_List)
data_analysis.plot(25, cumulative=False)

data_analysis = nltk.FreqDist(Tags_List)
data_analysis.plot(25, cumulative=False)

# Plot sentence by lenght
plt.hist([len(s) for s in Conversations_List], bins=50)
plt.title('Token per sentence')
plt.xlabel('Len (number of token)')
plt.ylabel('# samples')
plt.show()

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
import gensim.downloader as api
twitter_model = api.load("glove-twitter-25")

for sentence_index in range(len(Text_Data)):
    sentence = Text_Data[sentence_index]
    sentence_vector = []

    for word_index in range(len(sentence)):
        word = sentence[word_index]
        # word_vector = word2vec_model.wv[word]
        word_vector = twitter_model[word]
        sentence_vector.append(word_vector)

    word2vec_features.append(sentence_vector)

word2vec_features = np.asarray(word2vec_features)
print("the shape of sentence embedding is ", word2vec_features.shape)

print(word2vec_features)

# pad the input sentence encoding
from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX_LEN = 10

padded_sentence_encoding = pad_sequences(word2vec_features, padding="post", truncating="post", maxlen=MAX_LEN)
print("padded sentence shape is ", padded_sentence_encoding.shape)

# prepare tagging embedding

label_vectors = []

for index in range(len(Tags_List)):

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

    label_vectors.append(tag_vector)

label_vectors = np.asarray(label_vectors) # shape is (890 * 12)

# prepare sequence data

num_timesteps = 3 # time step of lstm
sequence_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(padded_sentence_encoding)-num_timesteps+1)] # get the index for each utterance senquence

sequence_data = []
sequence_target = []

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

    temp.append(label_vectors[i[0][0]])
    temp.append(label_vectors[i[1][0]])
    temp.append(label_vectors[i[2][0]])
    # temp.append(label_vectors[i[3][0]])
    # temp.append(label_vectors[i[4][0]])

    sequence_target.append(temp)

sequence_data = np.asarray(sequence_data) # (890, 5, [number of words in that sentence])
sequence_target = np.asarray(sequence_target) # (890, 5, 9)

print(sequence_data.shape)
print(sequence_target.shape)

from keras.models import Sequential
from keras.layers import TimeDistributed, LSTM, Dense, Bidirectional
from keras import regularizers
from keras.optimizers import Adam, RMSprop
from keras_contrib.layers import CRF
from sklearn.metrics import classification_report
from keras.utils.vis_utils import plot_model

# define utterance embedding layer
In_SEN_DIM = sequence_data.shape[3]
OUT_SEN_DIM = 128
print("input sentence dim is ", MAX_LEN, "*", In_SEN_DIM)
print("output sentence dim is ", OUT_SEN_DIM)

utter_embedding = Sequential()
utter_embedding.add(LSTM(OUT_SEN_DIM, return_sequences=False, input_shape=(MAX_LEN, In_SEN_DIM)))
utter_embedding.summary()

# define DA classification model
n_classes = 9

sequential_model = Sequential()
sequential_model.add(TimeDistributed(utter_embedding, input_shape=(num_timesteps, MAX_LEN, In_SEN_DIM)))
sequential_model.add(Bidirectional(LSTM(128, return_sequences=True, activation="relu")))
sequential_model.add(Bidirectional(LSTM(64, return_sequences=True, activation="relu")))
sequential_model.add(Dense(n_classes, activation="softmax", kernel_regularizer=regularizers.l2(0.01)))
# sequential_model.add(CRF(n_classes))
# sequential_model.compile(loss=CRF(n_classes).loss_function, optimizer="rmsprop", metrics=[CRF(n_classes).accuracy])
sequential_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.2, decay=0.001), metrics=['accuracy'])
sequential_model.summary()

plot_model(sequential_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# start training

sequential_model.fit(
    sequence_data,
    sequence_target,
    batch_size = 32,
    epochs = 20
)

y_pred = sequential_model.predict(
    sequence_data,
    verbose=0
)

y_pred_bool = np.argmax(y_pred, axis=1)
test_label_bool = np.argmax(sequence_target, axis=1)
print(classification_report(test_label_bool, y_pred_bool))

