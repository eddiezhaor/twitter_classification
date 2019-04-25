#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SYS6016
# NLP - LemonMonster
# Jiangxue Han, Jing Sun, Luke Kang, Runhao Zhao
# Baseline.py

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score     
from Preprocessing import PreprocessingData as p
import pandas as pd
import numpy as np   
class baselineModel:
    def tfidf(self,data,maxDf=0.1,minDF=0.001,testSize=0.3,randomState=42):
        v = TfidfVectorizer(analyzer = 'word',smooth_idf =True, min_df=minDF,max_df=maxDf)
        #split data into training and test set
        x_train, x_test, y_train, y_test = train_test_split(data.text, data.Prediction.values, test_size=testSize, random_state=randomState)
        v.fit(x_train)
        #tfidf transform
        x_train_tfidf = v.transform(x_train).toarray()
        x_train_tfidf = x_train_tfidf.astype(np.float)
        x_test_tfidf = v.transform(x_test).toarray()
        x_test_tfidf = x_test_tfidf.astype(np.float)
        #relabel -1 to 0
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0
        return x_train_tfidf, x_test_tfidf,y_train, y_test
    def naive_bayes(self,x_train,x_test,y_train,y_test,alpha=1):
        #create a naive bayes model
        clf = MultinomialNB(alpha=1)
        #fit on the training set
        clf.fit(x_train, y_train)
        #make predictions
        y_pred = clf.predict(x_test)
        #get the prediction accuracy
        score = accuracy_score(y_test, y_pred)
        return score, y_pred
        #PCA
    def pcaData(self,x_train,x_test,component=800):
        pca = PCA(n_components=component, svd_solver='full') 
        #fit on the training set   
        pca.fit(x_train) 
        #transformation
        new_train = pca.transform(x_train) 
        new_test = pca.transform(x_test )
        return new_train,new_test