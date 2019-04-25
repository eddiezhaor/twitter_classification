#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SYS6016
# NLP - LemonMonster
# Jiangxue Han, Jing Sun, Luke Kang, Runhao Zhao
# CNN_Class.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
class Cnn:
    def buildCnn(self,x_train,x_test,y_train,y_test):
        y_train = y_train.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        tf.reset_default_graph()
        n_inputs = x_train.shape[1]
        n_outputs = 1
        n_hidden1 = 500
        n_hidden2 = 300
        n_hidden3 = 100
        n_hidden4 = 50
        batch_size = 64
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.float32, shape=(None,n_outputs), name="y")
        #regularizer
        regularizer = tf.contrib.layers.l1_l2_regularizer()
        with tf.name_scope("dnn"):
            s = tf.shape(X)[0]
            X1 = tf.reshape(X,shape=[s,n_inputs,1])
            #the first layer
            conv1 = tf.layers.conv1d(X1, filters=4, kernel_size=10,strides=2,name="hidden1",activation=tf.nn.relu,activity_regularizer=regularizer)
            fc1 = tf.contrib.layers.flatten(conv1)
            logits = tf.layers.dense(fc1, n_outputs, name="outputs")
        
        
        with tf.name_scope("loss"):
            #entropy
            xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            #loss function
            loss = tf.reduce_mean(xentropy, name="loss")
            loss_summary = tf.summary.scalar('log_loss', loss)
            lr = 0.001
                
        
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999,\
                                                epsilon=1e-5)
            training_op = optimizer.minimize(loss)
        
        with tf.name_scope("eval"):
            predicted = tf.sigmoid(logits)
            correct_pred = tf.equal(tf.round(predicted), y)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
            
        init = tf.global_variables_initializer()
        p = 1
        n_epochs = 40
        writer = tf.summary.FileWriter('./graphs/train_cnn', tf.get_default_graph())
        test_writer = tf.summary.FileWriter('./graphs/test_cnn', tf.get_default_graph())
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                #batch
                for i in range(x_train.shape[0] // batch_size):
                    X_batch = x_train[i*batch_size:(i+1)*batch_size]
                    y_batch = y_train[i*batch_size:(i+1)*batch_size]
                    sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
                m,n = sess.run([accuracy,loss],feed_dict = {X:x_train, y:y_train})
                m1,n1 = sess.run([accuracy,loss],feed_dict = {X:x_test, y:y_test})
                print(epoch, "Train accuracy:", m, "Val accuracy:", m1)

env FLASK_APP=~/Desktop/ds5559/Project/polo2/app/app.py
env POLO_PROJ=~/Desktop/ds5559/Project/my_project/myproject