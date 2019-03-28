# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("./3y_Tokenized.csv",encoding='latin-1')
data.rename(columns={"Pre-processed":"text"},inplace=True)
texts = data.text.to_string()
#filter null
df = data[data['text'].notnull()]
#lowercase
df.text = df.text.apply(lambda x: " ".join(x.lower() for x in x.split()))
#remove punctuation
df['text'] = df['text'].str.replace('[^\w\s]','')
df['text'] = df['text'].str.replace('\\brt\\b\s\w+','')
#create new col for "num" count
df["num_count"] = df.text.str.count("num")
df['text'] = df['text'].str.replace('\\bnum\\b','')
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


from sklearn.feature_extraction.text import TfidfVectorizer
def tokenize(text):
    tokens = word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(item)
    return stems
v = TfidfVectorizer(tokenizer=tokenize, analyzer = 'word',smooth_idf =True, min_df=0.001,max_df=0.1)



from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(df_final.text, df_final.Prediction.values, test_size=0.3, random_state=42)
v.fit(x_train)
x_train_tfidf = v.transform(x_train).toarray()
x_train_tfidf = x_train_tfidf.astype(np.float)
x_test_tfidf = v.transform(x_test).toarray()
x_test_tfidf = x_test_tfidf.astype(np.float)

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

clf = MultinomialNB(alpha=1)
clf.fit(x_train_tfidf, y_train)
y_pred = clf.predict(x_test_tfidf)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


tf.reset_default_graph()

n_inputs = 1462
n_outputs = 2
n_hidden1 = 800
n_hidden2 = 500
n_hidden3 = 300
n_hidden4 = 100
n_hidden5 = 50

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
regularizer = tf.contrib.layers.l1_regularizer(scale=0.1)
regularizer = tf.contrib.layers.l1_regularizer(scale=0.1)
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu, kernel_regularizer = regularizer)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.elu, kernel_regularizer = regularizer)
    hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3",
                              activation=tf.nn.relu, kernel_regularizer = regularizer)
    hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4",
                              activation=tf.nn.elu, kernel_regularizer = regularizer)
    hidden5 = tf.layers.dense(hidden4, n_hidden5, name="hidden5",
                              activation=tf.nn.relu, kernel_regularizer = regularizer)
    logits = tf.layers.dense(hidden5, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)
    
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #optimizer = tf.train.MomentumOptimizer(learning_rate = 0.2, momentum=0.6)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999,\
    #                                   epsilon=1e-8)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    
init = tf.global_variables_initializer()

n_epochs = 10
batch_size = 128

with tf.Session() as sess:
    init.run()
    val_accuracy = []
    for epoch in range(n_epochs):
        for i in range(x_train_tfidf.shape[0] // batch_size):
            X_batch = x_train_tfidf[i*batch_size:(i+1)*batch_size]
            y_batch = y_train[i*batch_size:(i+1)*batch_size]
            sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})
        acc_val = accuracy.eval(feed_dict= {X: x_test_tfidf,y: y_test})
        val_accuracy.append(acc_val)
        print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
    Z = logits.eval(feed_dict = {X: x_test_tfidf})
    y_pred = np.argmax(Z, axis = 1)


