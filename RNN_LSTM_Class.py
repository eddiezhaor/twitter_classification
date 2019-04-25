#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SYS6016
# NLP - LemonMonster
# Jiangxue Han, Jing Sun, Luke Kang, Runhao Zhao
# FFNN_Class.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
class TextRNN:
    def rnn(self,X_train,X_valid,y_train,y_valid):
        y_train = y_train.reshape(-1,1)
        y_valid = y_valid.reshape(-1,1)
        y_train[y_train == -1] = 0
        y_valid[y_valid == -1] = 0
        
        tf.reset_default_graph()
               
        n_emb = 50
        n_steps = 28
        n_outputs = 1
        n_layers = 1
        n_neurons = 128
        
        X = tf.placeholder(tf.float32, shape=(None, n_steps, n_emb), name="X")
        y = tf.placeholder(tf.float32, shape=(None, n_outputs), name="y")
        
           
        with tf.name_scope("rnn"):
            lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons)
                          for layer in range(n_layers)]
            multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
            outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
            top_layer_h_state = states[-1][1]
            logits = tf.layers.dense(top_layer_h_state, n_outputs, name="outputs")
        
        
        
        with tf.name_scope("loss"):
            #entropy
            xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            #loss function
            loss = tf.reduce_mean(xentropy, name="loss")
            loss_summary = tf.summary.scalar('log_loss', loss)
        
        
        with tf.name_scope("train"):
            lr = 0.1
            #optimizer = tf.train.AdamOptimizer()
            #optimizer = tf.train.GradientDescentOptimizer(lr)
            #optimizer = tf.train.MomentumOptimizer(learning_rate = lr, momentum=0.8)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
            training_op = optimizer.minimize(loss)
        
        
        
        with tf.name_scope("eval"):
            predicted = tf.sigmoid(logits)
            correct_pred = tf.equal(tf.round(predicted), y)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
            accuracy_summary = tf.summary.scalar('accuracy', accuracy)
              
        init = tf.global_variables_initializer()
        
        n_epochs = 5
        batch_size = 64

        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                for i in range(X_train.shape[0] // batch_size):
                    X_batch = X_train[i*batch_size:(i+1)*batch_size]
                    #X_batch = X_batch.reshape((batch_size, n_steps, n_emb))
                    y_batch = y_train[i*batch_size:(i+1)*batch_size]
                    sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
                acc_train = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})
                acc_val = accuracy.eval(feed_dict= {X: X_valid,y: y_valid})
                print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
                