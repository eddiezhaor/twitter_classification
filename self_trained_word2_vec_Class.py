from gensim.models import Word2Vec

class my_word_vec:
    def train_word_vec(self,df):
        word_list = df.new_text.values.tolist()
        my_model = Word2Vec(df.new_text.values.tolist(),size=25,window=4,sg=0,min_count=1,workers=4)
        new_word_list=[]
        for i in word_list:
            l = 28 - len(i)
            new_word_list.append(i+["PAD"]*l)
        padding = np.random.randn(25)
        word_vec=[]
        new_word_1=[]
        for i in new_word_list:
            word_vec.append([my_model[x] if x !="PAD" else padding for x in i ])
        word_array = []
        for i in range(len(word_vec)):
            word_array.append(np.stack(word_vec[i],axis=1))
        word_array = np.array(word_array)
        word_array_1 = np.transpose(word_array,(0,2,1))
        del new_word_list
        del word_vec
        del word_array
        del my_model
        return word_array_1

    def train(self,df,word_array_1):
        X_train, X_valid, y_train, y_valid = train_test_split(word_array_1, df.Prediction, test_size=0.3,random_state=32)
        del word_array_1
        # del data
        # del df
        y_train = np.array(y_train)
        y_valid = np.array(y_valid)
        X_train = X_train.reshape(-1,28*25)
        X_valid = X_valid.reshape(-1,28*25)
        y_train = y_train.reshape(-1,1)
        y_valid = y_valid.reshape(-1,1)
        y_train[y_train == -1] = 0
        y_valid[y_valid == -1] = 0
        tf.reset_default_graph()
        n_inputs = X_train.shape[1]
        width = 25
        height = 28
        channels = 1
        n_outputs = 1
        batch_size = 500
        pool3_dropout_rate = 0.3
        fc1_dropout_rate = 0.5
        n_fc1 = 20
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.float32, shape=(None,n_outputs), name="y")
        #regularizer
        regularizer = tf.contrib.layers.l1_l2_regularizer()
        with tf.name_scope("dnn"):
        #     s = tf.shape(X)[0]
            X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
            training = tf.placeholder_with_default(False, shape=[], name='training') 
            y = tf.placeholder(tf.float32, shape=[None,1], name="y")  

        #the first layer
        conv1 = tf.layers.conv2d(X_reshaped, filters=64, kernel_size=(3,3),
                                strides=2, padding="VALID",
                                activation=tf.nn.relu, name="conv1")
        # conv2 = tf.layers.conv2d(X_reshaped, filters=64, kernel_size=3,kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                          strides=2, padding="VALID",
        #                          activation=tf.nn.relu, name="conv2")
        # Step 4: Set up the pooling layer with dropout using tf.nn.max_pool 
        # with tf.device("/gpu:0"):
        #     with tf.name_scope("pool3"):
        #         pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 5, 5, 1], padding="VALID")
        #         pool3_flat = tf.contrib.layers.flatten(pool3)
        #         pool3_flat_drop = tf.layers.dropout(pool3_flat, pool3_dropout_rate, training=training)

        # Step 5: Set up the fully connected layer using tf.layers.dense
        with tf.name_scope("fc1"):
            flatten = tf.contrib.layers.flatten(conv1)
        #     fc1 = tf.layers.dense(flatten, n_fc1, activation=tf.nn.relu, name="fc1")
        #     fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)

        #     fc1 = tf.contrib.layers.flatten(conv1)
        logits = tf.layers.dense(flatten, n_outputs, name="outputs")


        with tf.name_scope("loss"):
            #entropy
            xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            #loss function
            loss = tf.reduce_mean(xentropy, name="loss")
            loss_summary = tf.summary.scalar('log_loss', loss)
            lr = 0.01

        with tf.name_scope("train"):
        #     optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999,\
        #                                         epsilon=1e-5)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            training_op = optimizer.minimize(loss)
            
        with tf.name_scope("eval"):
            predicted = tf.sigmoid(logits)
            correct_pred = tf.equal(tf.round(predicted), y)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
            accuracy_summary = tf.summary.scalar('accuracy', accuracy)
        init = tf.global_variables_initializer()
        p = 1
        n_epochs = 80
        writer = tf.summary.FileWriter('./graphs/train_cnn', tf.get_default_graph())
        test_writer = tf.summary.FileWriter('./graphs/test_cnn', tf.get_default_graph())
        saver = tf.train.Saver()
        check_interval = 500
        best_loss_val = np.infty
        checks_since_last_progress = 0
        max_checks_without_progress = 20
        best_model_params = None 
        iteration = 0
        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                #batch
                for i in range(X_train.shape[0] // batch_size):
                    iteration += 1   
                    X_batch = X_train[i*batch_size:(i+1)*batch_size]
                    y_batch = y_train[i*batch_size:(i+1)*batch_size]
                    sess.run(training_op, feed_dict={X: X_batch, y: y_batch}) 
        #             if iteration % check_interval == 0:
        #                 loss_val = loss.eval(feed_dict={X: X_valid, y: y_valid})
        #                 if loss_val < best_loss_val:
        #                     best_loss_val = loss_val
        #                     checks_since_last_progress = 0
        #                 else:
        #                     checks_since_last_progress += 1
                m,n = sess.run([accuracy,loss],feed_dict = {X:X_batch, y:y_batch})
                m1,n1 = sess.run([accuracy,loss],feed_dict = {X:X_valid, y:y_valid})
                print(epoch, "Train accuracy:", m, "Val accuracy:", m1,"best_loss", best_loss_val)  
                m,n = sess.run([accuracy_summary,loss_summary],feed_dict = {X:X_batch, y:y_batch})
                m1,n1 = sess.run([accuracy_summary,loss_summary],feed_dict = {X:X_valid, y:y_valid})
                writer.add_summary(m,epoch)
                writer.add_summary(n,epoch)
                test_writer.add_summary(m1,epoch)
                test_writer.add_summary(n1,epoch)
                writer.close()
                test_writer.close()