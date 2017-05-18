import tensorflow as tf


class TextRNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size, rnn_hidden_size, rnn_num_layers, max_token_length, char_vocab_size,
            embedding_size, margin, num_neurons_fc, num_neurons_fc_2, loss_function, embedding_dim_char, filter_sizes_char, num_filters_char):

        def last_relevant(output, length):
            batch_size = tf.shape(output)[0]
            max_length = tf.shape(output)[1]
            out_size = int(output.get_shape()[2])
            index = tf.range(0, batch_size) * max_length + (length - 1)
            flat = tf.reshape(output, [-1, out_size])
            relevant = tf.gather(flat, index)
            return relevant

        # Placeholders for input, output and dropout
        self.neg_ind = tf.placeholder(tf.int32, [None], name="neg_ind")
        self.pos_ind = tf.placeholder(tf.int32, [None], name="pos_ind")
        self.increasing = tf.placeholder(tf.float32, [None, sequence_length], name="incr")
        self.input_x_q = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_q")
        self.input_x_t = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_t")
        self.input_x_q_len = tf.placeholder(tf.int32, [None], name="input_x_q_len")
        self.input_x_t_len = tf.placeholder(tf.int32, [None], name="input_x_t_len")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.input_x_q_char = tf.placeholder(tf.int32, [None, sequence_length, max_token_length], name="input_x_q_char")
        self.input_x_t_char = tf.placeholder(tf.int32, [None, sequence_length, max_token_length], name="input_x_t_char")
        extra_features = 5
        self.features = tf.placeholder(tf.float32, [None, extra_features], name="features")

        def calculate_loss(scores, pos_ind, neg_ind):
            if loss_function == "crossentropy":
                # CalculateMean cross-entropy loss
                losses = tf.nn.softmax_cross_entropy_with_logits(scores, self.input_y)
                return tf.reduce_mean(losses)
            elif loss_function == "maxmargin":
                # Calculate Max-Margin Loss
                scores_pos = tf.gather(scores, pos_ind)
                scores_neg = tf.gather(scores, neg_ind)

                scores_pos = tf.expand_dims(scores_pos, -1)
                scores_neg = tf.expand_dims(scores_neg, 0)

                # get negative scores, and calculate loss matrix with a margin of 1
                loss_matrix = tf.maximum(0., margin - scores_pos + scores_neg)  # we could also use tf.nn.relu here
                return tf.reduce_mean(loss_matrix)

                # Embedding layerx

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.W_char = tf.Variable(tf.random_uniform([char_vocab_size, embedding_dim_char], -1.0, 1.0),
                                      name="W_char")

            self.embedded_words_q = tf.nn.embedding_lookup(self.W, self.input_x_q)
            self.embedded_words_t = tf.nn.embedding_lookup(self.W, self.input_x_t)

            embedded_chars_q = tf.reshape(self.input_x_q_char, [-1, sequence_length * max_token_length])
            embedded_chars_q = tf.nn.embedding_lookup(self.W_char, embedded_chars_q)
            embedded_chars_q = tf.reshape(embedded_chars_q,
                                          [-1, sequence_length, max_token_length * embedding_dim_char])
            embedded_chars_q = tf.expand_dims(embedded_chars_q, -1)

            embedded_chars_t = tf.reshape(self.input_x_t_char, [-1, sequence_length * max_token_length])
            embedded_chars_t = tf.nn.embedding_lookup(self.W_char, embedded_chars_t)
            embedded_chars_t = tf.reshape(embedded_chars_t,
                                          [-1, sequence_length, max_token_length * embedding_dim_char])
            embedded_chars_t = tf.expand_dims(embedded_chars_t, -1)

            # This line adds a dimension just because of the channel
            self.embedded_words_expanded_q = tf.expand_dims(self.embedded_words_q, -1)
            self.embedded_words_expanded_t = tf.expand_dims(self.embedded_words_t, -1)

        pooled_outputs_q = []
        for i, filter_size in enumerate(filter_sizes_char):
            with tf.name_scope("conv-maxpool-char-%s" % filter_size):
                # Convolution Layer
                # We have 2 channels (question, tree-question)
                # [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [1, embedding_dim_char * filter_size, 1, num_filters_char]
                W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_filter_t")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters_char]), name="b_t")
                conv = tf.nn.conv2d(
                    embedded_chars_q,
                    W_filter,
                    strides=[1, 1, embedding_dim_char, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 1, max_token_length - filter_size + 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_q.append(pooled)

        pooled_outputs_t = []
        for i, filter_size in enumerate(filter_sizes_char):
            with tf.name_scope("conv-maxpool-char-%s" % filter_size):
                # Convolution Layer
                # We have 2 channels (question, tree-question)
                # [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [1, embedding_dim_char * filter_size, 1, num_filters_char]
                W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_filter_t")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters_char]), name="b_t")
                conv = tf.nn.conv2d(
                    embedded_chars_t,
                    W_filter,
                    strides=[1, 1, embedding_dim_char, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 1, max_token_length - filter_size + 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_t.append(pooled)

        # Combine all the pooled features
        num_filters_total_char = num_filters_char * len(filter_sizes_char)
        h_pool_char_q = tf.concat(3, pooled_outputs_q)
        h_pool_char_q = tf.reshape(h_pool_char_q, [-1, sequence_length, num_filters_total_char, 1])

        h_pool_char_t = tf.concat(3, pooled_outputs_t)
        h_pool_char_t = tf.reshape(h_pool_char_t, [-1, sequence_length, num_filters_total_char, 1])

        inputs_q = tf.concat(2, [self.embedded_words_expanded_q, h_pool_char_q])
        inputs_t = tf.concat(2, [self.embedded_words_expanded_t, h_pool_char_t])

        embedding_size += num_filters_total_char

        inputs_q = tf.reshape(inputs_q, [-1, sequence_length, embedding_size])
        inputs_t = tf.reshape(inputs_t, [-1, sequence_length, embedding_size])

        bidirectional = False
        if not bidirectional:
            with tf.variable_scope("rnn_q"):
                lstm_cell_q = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_size, state_is_tuple=True)
                lstm_cell_q = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_q, output_keep_prob=self.dropout_keep_prob)
                cell_q = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_q] * rnn_num_layers, state_is_tuple=True)
                outputs_q, _ = tf.nn.dynamic_rnn(cell_q, inputs_q, dtype=tf.float32, sequence_length=self.input_x_q_len)
            with tf.variable_scope("rnn_t"):
                lstm_cell_t = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_size, state_is_tuple=True)
                lstm_cell_t = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_t, output_keep_prob=self.dropout_keep_prob)
                cell_t = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_t] * rnn_num_layers, state_is_tuple=True)
                outputs_t, _ = tf.nn.dynamic_rnn(cell_t, inputs_t, dtype=tf.float32, sequence_length=self.input_x_t_len)
            last_q = last_relevant(outputs_q, self.input_x_q_len)
            last_t = last_relevant(outputs_t, self.input_x_t_len)
        else:
            # Do bidirectional rnn
            with tf.variable_scope("bidir_rnn_q"):
                lstm_cell_q = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_size, state_is_tuple=True)
                lstm_cell_q = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_q, output_keep_prob=self.dropout_keep_prob)
                cell_q = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_q] * rnn_num_layers, state_is_tuple=True)
                outputs_q, _ = tf.nn.bidirectional_dynamic_rnn(cell_q, cell_q, inputs_q, dtype=tf.float32, sequence_length=self.input_x_q_len)
                outputs_q_fw, outputs_q_bw = outputs_q
                outputs_q = tf.concat(2, [outputs_q_fw, outputs_q_bw])
                last_q = last_relevant(outputs_q, self.input_x_q_len)
            with tf.variable_scope("bidir_rnn_t"):
                lstm_cell_t = tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden_size, state_is_tuple=True)
                lstm_cell_t = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_t, output_keep_prob=self.dropout_keep_prob)
                cell_t = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_t] * rnn_num_layers, state_is_tuple=True)
                outputs_t, _ = tf.nn.bidirectional_dynamic_rnn(cell_t, cell_t, inputs_t, dtype=tf.float32, sequence_length=self.input_x_t_len)
                outputs_t_fw, outputs_t_bw = outputs_t
                outputs_t = tf.concat(2, [outputs_t_fw, outputs_t_bw])
                last_t = last_relevant(outputs_t, self.input_x_t_len)

        self.sims = tf.reduce_sum(tf.mul(last_q, last_t), 1, keep_dims=True)

        # Make input for classification
        self.new_input = tf.concat(1, [last_q, self.sims, last_t, self.features], name='new_input')

        multiplier = 4 if bidirectional else 2

        # hidden layer
        with tf.name_scope("hidden"):
            W = tf.get_variable(
                "W_hidden",
                shape=[multiplier * rnn_hidden_size + 1 + extra_features, num_neurons_fc],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_neurons_fc]), name="b")
            self.hidden_output = tf.nn.elu(tf.nn.xw_plus_b(self.new_input, W, b, name="hidden_output"))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")

        # hidden layer
        with tf.name_scope("hidden2"):
            W = tf.get_variable(
                "W_hidden2",
                shape=[num_neurons_fc, num_neurons_fc_2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_neurons_fc_2]), name="b")
            self.hidden_output_2 = tf.nn.elu(tf.nn.xw_plus_b(self.h_drop, W, b, name="hidden_output"))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop_2 = tf.nn.dropout(self.hidden_output_2, self.dropout_keep_prob, name="hidden_output_drop")

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_neurons_fc, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop_2, W, b, name="scores")

        with tf.device('/cpu:0'):
            with tf.name_scope("loss"):
                main_loss = calculate_loss(self.scores, self.pos_ind, self.neg_ind)
                self.loss = main_loss



