#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SYS6016
# NLP - LemonMonster
# Jiangxue Han, Jing Sun, Luke Kang, Runhao Zhao
# Preprocessing.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")
class PreprocessingData:
    def importData(self,filepath):
        data = pd.read_csv(filepath,encoding='latin-1')
        data.rename(columns={"Pre-processed":"text"},inplace=True)
        return data
    def clean(self,data):
        texts = data.text.to_string()
        #filter null
        df = data[data['text'].notnull()]
        #lowercase
        df.text = df.text.apply(lambda x: " ".join(x.lower() for x in x.split()))
        #remove punctuation
        df['text'] = df['text'].str.replace('<num>','')
        df['text'] = df['text'].str.replace('[^\w\s]','')
        df['text'] = df['text'].str.replace('\\brt\\b\s\w+','')
        df.text = df.text.apply(lambda x:" ".join(x.strip() for x in x.split()))
        #lemma
        lemma =WordNetLemmatizer()
        df['text'] = df['text'].apply(lambda x: " ".join([lemma.lemmatize(word) for word in x.split()]))
        #remove stop words
        stop = stopwords.words('english')
        extra_stopwords = """
        us rest went least would much must long one like much say well without though yet might still upon
        done every rather particular made many previous always never thy thou go first oh thee ere ye came
        almost could may sometimes seem called among another also however nevertheless even way one two three
        ever put
        """.strip().split()
        df['text'] = df['text'].apply(lambda x: " ".join(word for word in x.split() if word not in stop))
        df['text'] = df['text'].apply(lambda x: " ".join(word for word in x.split() if word not in extra_stopwords))
        #Drop duplicates
        df.drop_duplicates(subset=["Prediction","text"],inplace=True)
        df_final = df[['text','Prediction']]
        return df_final
        