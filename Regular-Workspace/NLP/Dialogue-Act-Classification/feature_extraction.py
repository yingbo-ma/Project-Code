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

# data_analysis = nltk.FreqDist(Speakers_List)
# data_analysis.plot(25, cumulative=False)
#
# data_analysis = nltk.FreqDist(Conversations_List)
# data_analysis.plot(25, cumulative=False)
#
# data_analysis = nltk.FreqDist(Tags_List)
# data_analysis.plot(25, cumulative=False)

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

# text normalization
for data in Text_Data:
    for word_index in range(len(data)):
        if data[word_index] == 's': # for " ..'s'.. "
            if data[word_index-1] == 'let': # let's..
                data[word_index] = 'us'
            else: # he's, it's, ..
                data[word_index] = 'is'

        if data[word_index] == 't': # for " ..'t'.. "
            data[word_index] = 'not'

        if data[word_index] == 'd': # for " ..'d'.. "
            data[word_index] = 'would'

        if data[word_index] == 'm': # for " ..'m'.. "
            data[word_index] = 'am'
    print(data)

# prepare feature 1: how many words in each utterance?
word_number_feature = []

for utterance_index in range(len(Text_Data)):
    utterance = Text_Data[utterance_index]
    word_number_feature.append(len(utterance))

# prepare feature 2: how many utterance this speaker already said?

utterance_spoken_feature = []

for speaker_index in range(len(Speakers_List)):
    if speaker_index == 0:
        utterance_spoken_feature.append(1)
    elif Speakers_List[speaker_index] == Speakers_List[speaker_index - 1]:
        utterance_spoken_feature.append(utterance_spoken_feature[-1] + 1)
    else: # there is a speaker switch
        utterance_spoken_feature.append(1)


# prepare feature 3: dialogue act tagging
da_feature = []

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

    da_feature.append(tag_vector)

# prepare feature 4: bag of words
word2count = {} # We declare a dictionary to hold our bag of words

for data in Text_Data:
    for word in data:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1

import heapq
freq_words = heapq.nlargest(100, word2count, key=word2count.get)

print(freq_words)
print(len(freq_words)) # 100
print(len(word2count)) # 1614

bag_of_word_feature = []
for data in Text_Data:
    vector = []
    for word in freq_words:
        if word in data:
            vector.append(1)
        else:
            vector.append(0)
    bag_of_word_feature.append(vector)

########  until now, we have four features, 1: word number(numerical) 2: utterance being said(numerical) 3: dialogue act tagging (catogorical) 4: utterance vector: bag of words (vector)  ############################################################

print(len(word_number_feature))
print(word_number_feature)

print(len(utterance_spoken_feature))
print(utterance_spoken_feature)

print(len(da_feature))
print(da_feature)
#
# print(len(bag_of_word_feature))
# print(bag_of_word_feature)

########  this is the prediction variable, speaker turn  ############################################
# prepare speaker-turn taggingg embedding
speaker_turn = [] # st: speaker-turn

for index in range(len(Speakers_List)):
    if index == len(Speakers_List) - 1:
        speaker_turn.append(0)
    elif Speakers_List[index] == Speakers_List[index + 1]:
        speaker_turn.append(0)
    else:
        speaker_turn.append(1)

print(len(speaker_turn))
print(speaker_turn)

######################################################################################################
# import seaborn as sns
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)
#
# data_analysis = nltk.FreqDist(speaker_turn)
# data_analysis.plot(25, cumulative=False)
# ax = sns.countplot(da_feature, x='speaker switch', palette='hls')
# plt.show()
######## create logistic model ###################################################################################
X = []
for feature_index in range(len(Text_Data)):
    feature_vector = []
    feature_vector.append(word_number_feature[feature_index])
    feature_vector.append(utterance_spoken_feature[feature_index])
    feature_vector.extend(da_feature[feature_index])
    feature_vector.extend(bag_of_word_feature[feature_index])
    # print(feature_vector)
    X.append(feature_vector)

X = np.asarray(X)
print(len(X)) # 8077
print(len(X[0])) # 111

X = X.T

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
pca.fit(X)
X = pca.components_
print(X)
print(len(X)) # 10
print(len(X[0])) # 8077

X = X.T

###### build the model ###############################################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# X = np.asarray(X)
Y = np.asarray(speaker_turn)

train_test_split = int(0.7*len(X))

X_train = X[0:train_test_split]
X_test = X[train_test_split:]

Y_train = Y[0:train_test_split]
Y_test = Y[train_test_split:]

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
print(classification_report(Y_test,predictions))

from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
print(classification_report(Y_test,predictions))

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
print(classification_report(Y_test,predictions))

# X = []
# for feature_index in range(len(Text_Data)):
#     # feature_vector = []
#     # # feature_vector.append(word_number_feature[feature_index])
#     # feature_vector.append(utterance_spoken_feature[feature_index])
#     # # feature_vector.append(da_feature[feature_index])
#     # # feature_vector.append(bag_of_word_feature[feature_index])
#     # X.append(feature_vector)
#     feature_vector = []
#     feature_vector.extend(word_number_feature[feature_index])
#     feature_vector.extend(utterance_spoken_feature[feature_index])
#     feature_vector.extend(da_feature[feature_index])
#     feature_vector.extend(bag_of_word_feature[feature_index])
#     X.append(feature_vector)
#
# print(X)
#
# ###### build the model ###############################################################
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix
#
# X = np.asarray(X)
# Y = np.asarray(speaker_turn)
#
# train_test_split = int(0.7*len(X))
#
# X_train = X[0:train_test_split]
# X_test = X[train_test_split:]
#
# Y_train = Y[0:train_test_split]
# Y_test = Y[train_test_split:]
#
# model = LogisticRegression(solver='liblinear', random_state=0)
# model.fit(X_train, Y_train)
# predictions = model.predict(X_test)
# print(classification_report(Y_test,predictions))
#
# from sklearn import svm
# clf = svm.SVC()
# clf.fit(X_train, Y_train)
# predictions = clf.predict(X_test)
# print(classification_report(Y_test,predictions))
#
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(X_train, Y_train)
# predictions = model.predict(X_test)
# print(classification_report(Y_test,predictions))

########## neural network #####################################################

# prepare sequence data

num_timesteps = 3 # time step of lstm
sequence_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(X)-num_timesteps+1)] # get the index for each utterance senquence

sequence_data = []
sequence_target = []

for i in sequence_index:
    temp = []

    temp.append(X[i[0][0]])
    temp.append(X[i[1][0]])
    temp.append(X[i[2][0]])
    # temp.append(word2vec_features[i[3][0]])
    # temp.append(word2vec_features[i[4][0]])

    sequence_data.append(temp)

for i in sequence_index:
    temp = []

    temp.append(Y[i[0]])
    temp.append(Y[i[1]])
    temp.append(Y[i[2]])
    # temp.append(word2vec_features[i[3][0]])
    # temp.append(word2vec_features[i[4][0]])

    sequence_target.append(temp)

sequence_data = np.asarray(sequence_data) # (890, 5, [number of words in that sentence])
sequence_target = np.asarray(sequence_target)

from keras.models import Sequential
from keras.layers import TimeDistributed, LSTM, Dense, Bidirectional
from keras import regularizers
from keras.optimizers import Adam, RMSprop
from keras_contrib.layers import CRF
from sklearn.metrics import classification_report
from keras.utils.vis_utils import plot_model

sequential_model = Sequential()
sequential_model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(num_timesteps, 100)))
sequential_model.add(TimeDistributed(Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01)))) # if the output label is a vector list, then should use time distributed Dense(1) !!!
sequential_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.2, decay=0.001), metrics=['accuracy'])
sequential_model.summary()

# start training
print("sequence_data", sequence_data.shape)
print("sequence_target", sequence_target.shape)

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
test_label_bool = np.argmax(Y, axis=1)
print(classification_report(test_label_bool, y_pred_bool))