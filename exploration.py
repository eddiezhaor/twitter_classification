# -*- coding: utf-8 -*-
import pandas as pd

data = pd.read_csv("3y_Tokenized.csv", encoding = "latin-1")  # (386251, 8)
data.shape
# (386251, 8)

#filter null
df = data[data['Pre-processed'].notnull()] # (386250, 8)
df = df.rename({"Pre-processed":"text"}, axis='columns')

#lowercase
df.text = df.text.apply(lambda x: " ".join(x.lower() for x in x.split()))
#remove punctuation
df['text'] = df['text'].str.replace('[^\w\s]','')
df['text'] = df['text'].str.replace('\\brt\\b\s\w+','')
df['text'] = df['text'].str.replace('\\bnum\\b','')
df.text = df.text.apply(lambda x:" ".join(x.strip() for x in x.split()))


#lemma
lemma =WordNetLemmatizer()
df['text'] = df['text'].apply(lambda x: " ".join([lemma.lemmatize(word) for word in x.split()]))
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(word for word in x.split() if x not in stop))
#lemma
lemma =WordNetLemmatizer()
df['text'] = df['text'].apply(lambda x: " ".join([lemma.lemmatize(word) for word in x.split()]))
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(word for word in x.split() if x not in stop))

raw = pd.read_csv("3y_Tokenized.csv", encoding = "latin-1")
head = raw.head()

import re
line = re.sub('[!@#$]', '', line)
                
line = head.iloc[0,0]