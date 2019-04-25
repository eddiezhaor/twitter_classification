#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SYS6016
# NLP - LemonMonster
# Jiangxue Han, Jing Sun, Luke Kang, Runhao Zhao
# Train.py

#import preprocessing
from Preprocessing import PreprocessingData as p
#import baseline model
from Baseline import baselineModel as b
#import ffnn
from FFNN_Class import FFNN as f
#import CNN
from CNN_Class import Cnn as c
z = p()
df = z.importData("./3y_Tokenized.csv")
finaldf = z.clean(df)
baseline = b()
#tf-idf and train-test split
x_train_tfidf, x_test_tfidf,y_train, y_test = baseline.tfidf(finaldf)
#pca
pca_train, pca_test = baseline.pcaData(x_train_tfidf,x_test_tfidf)
#naive bayes
score, pred = baseline.naive_bayes(x_train_tfidf, x_test_tfidf,y_train, y_test)
score
#train a ffnn model
ffnn = f()
ffnn.fnn(x_train_tfidf, x_test_tfidf,y_train, y_test)

#train a cnn model
cnn = c()
cnn.buildCnn(x_train_tfidf, x_test_tfidf,y_train, y_test)
