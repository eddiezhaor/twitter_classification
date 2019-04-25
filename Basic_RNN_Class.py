class basic_rnn:
    def b_rnn(self,a,df):
        #RNN
        X_train, X_valid, y_train, y_valid = train_test_split(a, df.Prediction, test_size=0.3,random_state=32)
        del data
        del df
        y_train = np.array(y_train)
        y_valid = np.array(y_valid)
        # X_train = X_train.reshape(-1,28*25)
        # X_valid = X_valid.reshape(-1,28*25)
        # y_train = y_train.reshape(-1,1)
        # y_valid = y_valid.reshape(-1,1)
        y_train[y_train == -1] = 0
        y_valid[y_valid == -1] = 0
        tf.reset_default_graph()

        n_steps = 28
        n_inputs = 50
        n_neurons = 100
        n_outputs = 2

        learning_rate = 0.001

        X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
        y = tf.placeholder(tf.int32, [None])

        basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
        outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

        logits = tf.layers.dense(states, n_outputs)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                logits=logits)
        loss = tf.reduce_mean(xentropy)
        loss_summary = tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,\
                                        epsilon=1e-8)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        batch_size = 200

        def shuffle_batch(X, y, batch_size):
            rnd_idx = np.random.permutation(len(X))
            n_batches = len(X) // batch_size
            for batch_idx in np.array_split(rnd_idx, n_batches):
                X_batch, y_batch = X[batch_idx], y[batch_idx]
                yield X_batch, y_batch

        # X_test = X_test.reshape((-1, n_steps, n_inputs))

        n_epochs = 20

        writer = tf.summary.FileWriter('./graphs/train2', tf.get_default_graph())
        test_writer = tf.summary.FileWriter('./graphs/test2', tf.get_default_graph())

        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                    X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                    sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                acc_test = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
                print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)
        #         X_train = X_train.reshape((-1, n_steps, n_inputs))
                m,n = sess.run([accuracy_summary,loss_summary],feed_dict = {X:X_train, y:y_train})
                m1,n1 = sess.run([accuracy_summary,loss_summary],feed_dict = {X:X_valid, y:y_valid})
                writer.add_summary(m,epoch)
                writer.add_summary(n,epoch)
                test_writer.add_summary(m1,epoch)
                test_writer.add_summary(n1,epoch)
        #         save_path = saver.save(sess, "./my_model_final.ckpt")
                
        # writer.close()
        # test_writer.close()