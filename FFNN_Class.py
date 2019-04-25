#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SYS6016
# NLP - LemonMonster
# Jiangxue Han, Jing Sun, Luke Kang, Runhao Zhao
# FFNN_Class.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
class FFNN:
    def fnn(self,x_train,x_test,y_train,y_test):
        y_train = y_train.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        tf.reset_default_graph()
        n_inputs = x_train.shape[1]
        n_outputs = 1
        n_hidden1 = 800
        n_hidden2 = 500
        n_hidden3 = 300
        n_hidden4 = 100
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.float32, shape=(None,n_outputs), name="y")
        regularizer = tf.contrib.layers.l1_l2_regularizer()
        with tf.name_scope("dnn"):
            #create first layer
            hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                                      activation=tf.nn.relu, kernel_regularizer = regularizer)
            #second hidden layer
            hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                                      activation=tf.nn.relu, kernel_regxularizer = regularizer)
            #third hidden layer
            hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3",
                                      activation=tf.nn.relu, kernel_regularizer = regularizer)
            hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4",
                                      activation=tf.nn.relu, kernel_regularizer = regularizer)
            logits = tf.layers.dense(hidden4, n_outputs, name="outputs")
        
        with tf.name_scope("loss"):
            xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(xentropy, name="loss")
            loss_summary = tf.summary.scalar('log_loss', loss)
            
        lr = 0.001
        
        with tf.name_scope("train"):
             optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999,\
                                                epsilon=1e-8)
             training_op = optimizer.minimize(loss)
        
        with tf.name_scope("eval"):
            predicted = tf.sigmoid(logits)
            correct_pred = tf.equal(tf.round(predicted), y)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))       
        init = tf.global_variables_initializer()
        
        n_epochs = 10
        batch_size = 60
        with tf.Session() as sess:
            init.run()
            val_accuracy = []
            for epoch in range(n_epochs):
                for i in range(x_train.shape[0] // batch_size):
                    X_batch = x_train[i*batch_size:(i+1)*batch_size]
                    y_batch = y_train[i*batch_size:(i+1)*batch_size]
                    sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
                acc_train = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})
                acc_val = accuracy.eval(feed_dict= {X: x_test,y: y_test})
                print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
            