import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
import string
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
warnings.filterwarnings("ignore")

class pretrained:
    def word_vev(self,df):
        weights = []
        PAD_TOKEN = 0
        word2idx = { 'PAD': PAD_TOKEN }
        with open('glove.6B.50d.txt',"r") as file: 
            for index, line in enumerate(file): 
                values = line.split() # Word and weights separated by space 
                word = values[0] # Word is first symbol on each line 
                word_weights = np.asarray(values[1:], dtype=np.float32) # Remainder of line is weights for word 
                word2idx[word] = index + 1 # PAD is our zeroth index so shift by one 
                weights.append(word_weights)
                if index + 1 == 40_000:
                # Limit vocabulary to top 40k terms
                    break

        embedding_dim = len(weights[0])
        weights.insert(0, np.random.randn(embedding_dim))

        unknown_token=len(weights) 
        word2idx['unk'] = unknown_token 
        weights.append(np.random.randn(embedding_dim))

        weights = np.asarray(weights, dtype=np.float32)
        vocab_size=weights.shape[0]

        import nltk
        features = {}
        feature_list = []
        ind = 0
        for i in df.new_text:
            features[ind] = i # ['hello', 'world']
            feature_list.append([word2idx.get(word, unknown_token) for word in features[ind]])
        new_feature= []
        for i in feature_list:
            l = 28 - len(i)
            new_feature.append(i+[0]*l)

        tf.reset_default_graph()
        glove_weights_initializer = tf.constant_initializer(weights)
        embedding_weights = tf.get_variable(
            name='embedding_weights', 
            shape=(vocab_size, embedding_dim), 
            initializer=glove_weights_initializer,trainable=False)
        embedding = tf.nn.embedding_lookup(embedding_weights,new_feature)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            a = sess.run(embedding)
        a = a
        return a