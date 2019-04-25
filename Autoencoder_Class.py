class Autoencoder(object):
    def __init__(self, inout_dim, encoded_dim):
        learning_rate = 0.1 
        hidden2 = inout_dim//2
        # Weights and biases
        hiddel_layer_weights = tf.Variable(tf.random_normal([inout_dim, hidden2]))
        hiddel_layer_biases = tf.Variable(tf.random_normal([hidden2]))
        hiddel_layer_weights_2 = tf.Variable(tf.random_normal([hidden2, encoded_dim]))
        hiddel_layer_biases_2 = tf.Variable(tf.random_normal([encoded_dim]))
        hiddel_layer_weights_3 = tf.Variable(tf.random_normal([encoded_dim, hidden2]))
        hiddel_layer_biases_3 = tf.Variable(tf.random_normal([hidden2]))
        output_layer_weights = tf.Variable(tf.random_normal([hidden2, inout_dim]))
        output_layer_biases = tf.Variable(tf.random_normal([inout_dim]))
        
        # Neural network
        self._input_layer = tf.placeholder(tf.float32, [None, inout_dim])
        self._hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(self._input_layer, hiddel_layer_weights), hiddel_layer_biases))
        self._hidden_layer = tf.nn.relu(tf.add(tf.matmul(self._hidden_layer_1, hiddel_layer_weights_2), hiddel_layer_biases_2))
        self._hidden_layer_2 = tf.nn.relu(tf.add(tf.matmul(self._hidden_layer, hiddel_layer_weights_3), hiddel_layer_biases_3))
        self._output_layer = tf.matmul(self._hidden_layer_2, output_layer_weights) + output_layer_biases
        self._real_output = tf.placeholder(tf.float32, [None, inout_dim])
        
        self._meansq = tf.sqrt(tf.reduce_mean(tf.square(self._output_layer - self._real_output)))
        self._optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self._meansq)
        self._training = tf.global_variables_initializer()
        self._session = tf.Session()
        
    def train(self, input_train, input_test, batch_size, epochs):
        self._session.run(self._training)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(int(input_train.shape[0]/batch_size)):
                epoch_input = input_train[ i * batch_size : (i + 1) * batch_size ]
                _, c = self._session.run([self._optimizer, self._meansq], feed_dict={self._input_layer: epoch_input, self._real_output: epoch_input})
            rmse = self._session.run(self._meansq,feed_dict={self._input_layer:input_train,self._real_output:input_train})
            print('Epoch', epoch, 'loss:',rmse)
        
    def getEncoded(self, image):
        encoded = self._session.run(self._hidden_layer, feed_dict={self._input_layer:image})
        return encoded
