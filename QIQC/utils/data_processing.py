# -*- coding = utf-8 -*-

# @author:黑白
# @contact:1808132036@qq.com
# @time:19-2-19下午9:12
# @file:data_processing.py

import os
import re
import numpy as np
import pandas as pd

# import spacy
# from spacy.lang.en.stop_words import STOP_WORDS


from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors

STOP_WORDS = []

DATA_PATH = os.getcwd() + os.sep + u'../data/'
EMB_PATH = os.getcwd() + os.sep + u'../embeddings'

TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')
TEST_PATH = os.path.join(DATA_PATH, 'test.csv')

GLOVE_PATH = os.path.join(EMB_PATH, 'glove.840B.300d', 'glove.840B.300d.txt')
glove_mean = -0.005838499
glove_std = 0.48782197

GOOGLENEWS_PATH = os.path.join(EMB_PATH, 'GoogleNews-vectors-negative300', 'GoogleNews-vectors-negative300.bin')
PARAGRAM_PATH = os.path.join(EMB_PATH, 'paragram_300_s1999', 'paragram_300_s1999.txt')
paragram_mean = -0.0053247944
paragram_std = 0.49346468

WIKI_NEWS = os.path.join(EMB_PATH, 'wiki-news-300d-1M', 'wiki-news-300d-1M.vec')

vocab_size = 120000
max_seq_len = 72
emb_size = 300


def clean_data(sequence):
    # 1. lower
    x = str(sequence)
    x = x.lower()

    # 2. punc data
    puncts = [
        ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
        '+', '\\', '•', '~', '@', '£',
        '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
        '½', 'à', '…',
        '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
        '▓', '—', '‹', '─',
        '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
        'Ã', '⋅', '‘', '∞',
        '∙', '）', '↓', '、', '│', '₹', 'π', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï',
        'Ø', '¹', '≤', '‡', '√']

    for punct in puncts:
        if punct in x:
            x = x.replace(punct, ' ')

    for punct in '&':
        x = x.replace(punct, f' {punct} ')

    # 3. number data
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    x = re.sub('[0-9]{1}', '#', x)

    # 4.mispell data
    mispell_dict = {
        "aren't": "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "I would",
        "i'd": "I had",
        "i'll": "I will",
        "i'm": "I am",
        "isn't": "is not",
        "it's": "it is",
        "it'll": "it will",
        "i've": "I have",
        "let's": "let us",
        "mightn't": "might not",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "we'd": "we would",
        "we're": "we are",
        "weren't": "were not",
        "we've": "we have",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
        "'re": " are",
        "wasn't": "was not",
        "we'll": " will",
        "tryin'": "trying",
        'colour': 'color',
        'centre': 'center',
        'didnt': 'did not',
        'doesnt': 'does not',
        'isnt': 'is not',
        'shouldnt': 'should not',
        'favourite': 'favorite',
        'travelling': 'traveling',
        'counselling': 'counseling',
        'theatre': 'theater',
        'cancelled': 'canceled',
        'labour': 'labor',
        'organisation': 'organization',
        'wwii': 'world war 2',
        'citicise': 'criticize',
        'instagram': 'social medium',
        'whatsapp': 'social medium',
        'snapchat': 'social medium'
    }
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    def replace(match):
        return mispell_dict[match.group(0)]
    x = mispell_re.sub(replace, x)

    # 5. stop word
    tmp = ""
    for word in x.split():
        if word not in list(STOP_WORDS):
            tmp += word + ' '

    x = tmp

    return x

# read data
def read_data():

    train_df = pd.read_csv(TRAIN_PATH, sep=',', encoding='utf-8').head(50000)
    test_df = pd.read_csv(TEST_PATH)

    train_x_len = []
    dev_x_len = []
    test_x_len = []

    #clean data
    train_df['question_text'] = train_df['question_text'].map(lambda x: clean_data(x))
    test_df['question_text'] = test_df['question_text'].map(lambda x: clean_data(x))

    # Fill NaN
    # train_x = train_df['question_text'].fillna('_na_').values
    # test_x = test_df['question_text'].fillna('_na_').values
    # train_y = np.array(train_df['target'])
    # train_y = [1 for _ in range(len(train_df['target'].values))]

    train_x = train_df['question_text']
    test_x = test_df['question_text']
    train_y = train_df['target']

    # print("train_y value: ")
    # for y_value in train_y:
    #     if y_value != 0 or y_value != 1:
    #         print(y_value)

    # split
    train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.9)

    # Tokenize the sentence
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(list(train_x))

    train_x = tokenizer.texts_to_sequences(train_x)
    dev_x = tokenizer.texts_to_sequences(dev_x)
    test_x = tokenizer.texts_to_sequences(test_x)

    for text in train_x:
        train_x_len.append(len(text))
    for text in dev_x:
        dev_x_len.append(len(text))
    for text in test_x:
        test_x_len.append(len(text))

    #Pad the sentencen
    train_x = pad_sequences(train_x, maxlen=max_seq_len)
    dev_x = pad_sequences(dev_x, maxlen=max_seq_len)
    test_x = pad_sequences(test_x, maxlen=max_seq_len)

    return train_x, train_y, dev_x, dev_y, test_x, train_x_len, dev_x_len, test_x_len, tokenizer.word_index

def load_embedding(file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    if file == GOOGLENEWS_PATH:
        embeddings_index = KeyedVectors.load_word2vec_format(file, binary=True)
    elif file == os.path.join(DATA_PATH, "embeddings", "wiki-news-300d-1M", "wiki-news-300d-1M.vec"):
        embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(file, encoding='latin'))
    return embeddings_index


def build_vocab(sentences, verbose=True):
    #tokenize the sentences
    tokenizer = Tokenizer(
        num_words=vocab_size,
        filters='!"#$%()*+,-./:;<=>?@[\]^_`{|}~ '
    )
    tokenizer.fit_on_texts(sentences)
    vocab = tokenizer.word_index
    return vocab

# Make embedding matrix
def make_embedding_matrixs(
        word_index_dict,
        embedding_index_dict,
        emb_mean=glove_mean, emb_std=glove_std,
        vocab_size=vocab_size, emb_size=emb_size):

    nb_words = min(vocab_size, len(word_index_dict))
    embedding_matrix=np.random.normal(emb_mean, emb_std,(nb_words+1, emb_size))

    for word, i in word_index_dict.items():
        if i >= vocab_size:
            continue
        embedding_vector = embedding_index_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


# Get Batch
def get_train_batch(train_x, train_y, train_x_len, batch_size, batch_num):
    start_index = batch_size * batch_num
    end_index = min(start_index+batch_size, len(train_x_len))
    return np.array(train_x[start_index:end_index]), np.array(train_y[start_index:end_index]).astype(np.int32), np.array(train_x_len[start_index:end_index])


#########################################################################

if __name__ == '__main__':
    train_x, train_y, dev_x, dev_y, test_x, train_x_len, dev_x_len, test_x_len, word_index = read_data()
    print("eee")
    for y in train_y[901:1500]:
        print(y)