import re
import nltk

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

# feature 1: Number of words in the utterance (NW)
# feature 2: Number of utterance a speaker already spoke (NU)
# feature 3: Dialogue Act Tagging of each utterance (DA)
# feature 4: Sentence Embedding (SE) (bag of words, word2vec)

def feature_NW(Text_Corpus):
    NW_feature = []
    corpus_size = len(Text_Corpus)
    for utterance_index in range(corpus_size):
        utterance = Text_Corpus[utterance_index]
        utterance_list = nltk.word_tokenize(utterance)
        utterance_length = len(utterance_list)
        NW_feature.append(utterance_length)
    return NW_feature

def feature_NU(Speakers_Corpus): # we use the before how many sentence spoken to predict whether is going to have speaker switch in this sentence or not
    NU_feature = []
    corpus_size = len(Speakers_Corpus)
    for speaker_index in range(corpus_size):
        if speaker_index == 0:
            NU_feature.append(1)
        elif Speakers_Corpus[speaker_index] == Speakers_Corpus[speaker_index-1]: # the speaker in this sentence not changed
            NU_feature.append(NU_feature[-1]+1)
        else: # the speaker in this sentence changed
            NU_feature.append(1)
    return NU_feature

def feature_SE(Text_Corpus):
    SE_feature = []

# feature: topic modelling

