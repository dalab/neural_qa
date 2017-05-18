import tensorflow as tf


class TextCNNChar(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling, 2 FC and a softmax layer.
    Architectures:
        (1) Dot product of sentence embeddings
        (2) Dot product with matrix of sentence embeddings
        (3) sentence embeddings concatenated, with 2 FC
        (4) sentence embeddings concatenated with matrix dot product, with 2 FC
        (5) sentence embeddings concatenated with 2 FC, combined in end with matrix dot product
    """

    def __init__(
            self, sequence_length, max_token_length, num_classes, batch_size, vocab_size, char_vocab_size, architecture,
            embedding_size, filter_sizes, num_filters, margin, l2_reg_lambda,
            num_neurons_fc, num_neurons_fc_2, target_lambda, loss_function, embedding_dim_char, filter_sizes_char, num_filters_char):

        # Placeholders for input, output and dropout
        self.neg_ind = tf.placeholder(tf.int32, [None], name="neg_ind")
        self.pos_ind = tf.placeholder(tf.int32, [None], name="pos_ind")

        extra_features = 5
        self.features = tf.placeholder(tf.float32, [None, extra_features], name="features")

        self.input_x_q = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_q")
        self.input_x_t = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_t")

        self.input_x_q_char = tf.placeholder(tf.int32, [None, sequence_length, max_token_length], name="input_x_q_char")
        self.input_x_t_char = tf.placeholder(tf.int32, [None, sequence_length, max_token_length], name="input_x_t_char")

        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_x_q_len = tf.placeholder(tf.int32, [None], name="input_x_q_len")
        self.input_x_t_len = tf.placeholder(tf.int32, [None], name="input_x_t_len")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

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

                # get negative scores, and calculate loss matrix with a margin
                loss_matrix = tf.maximum(0., margin - scores_pos + scores_neg)
                return tf.reduce_mean(loss_matrix)

        # Embedding layerx
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.W_char = tf.Variable(tf.random_uniform([char_vocab_size, embedding_dim_char], -1.0, 1.0), name="W_char")

            self.embedded_words_q = tf.nn.embedding_lookup(self.W, self.input_x_q)
            self.embedded_words_t = tf.nn.embedding_lookup(self.W, self.input_x_t)

            embedded_chars_q = tf.reshape(self.input_x_q_char, [-1, sequence_length * max_token_length])
            embedded_chars_q = tf.nn.embedding_lookup(self.W_char, embedded_chars_q)
            embedded_chars_q = tf.reshape(embedded_chars_q, [-1, sequence_length, max_token_length * embedding_dim_char])
            embedded_chars_q = tf.expand_dims(embedded_chars_q, -1)

            embedded_chars_t = tf.reshape(self.input_x_t_char, [-1, sequence_length * max_token_length])
            embedded_chars_t = tf.nn.embedding_lookup(self.W_char, embedded_chars_t)
            embedded_chars_t = tf.reshape(embedded_chars_t, [-1, sequence_length, max_token_length * embedding_dim_char])
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

        self.embedded_words_expanded_q = tf.concat(2, [self.embedded_words_expanded_q, h_pool_char_q])
        self.embedded_words_expanded_t = tf.concat(2, [self.embedded_words_expanded_t, h_pool_char_t])

        embedding_size += num_filters_total_char

        '''
            THIS ITS SAME MODEL AS TEXT_CNN (EXCEPT EMBEDDING DIM SIZE)'
        '''

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_q = []
        pooled_outputs_t = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-q-%s" % filter_size):
                # Convolution Layer
                # We have 2 channels (question, tree-question)
                # [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_filter_q")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_q")
                conv = tf.nn.conv2d(
                    self.embedded_words_expanded_q,
                    W_filter,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_q.append(pooled)

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-t-%s" % filter_size):
                # Convolution Layer
                # We have 2 channels (question, tree-question)
                # [filter_height, filter_width, in_channels, out_channels]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_filter_t")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_t")
                conv = tf.nn.conv2d(
                    self.embedded_words_expanded_t,
                    W_filter,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_t.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool_q = tf.concat(3, pooled_outputs_q)
        h_pool_t = tf.concat(3, pooled_outputs_t)
        self.h_pool_flat_q = tf.reshape(h_pool_q, [-1, num_filters_total])
        self.h_pool_flat_t = tf.reshape(h_pool_t, [-1, num_filters_total])

        # Keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        # Compute similarity
        with tf.name_scope("similarity"):
            W = tf.get_variable(
                "W_sim",
                shape=[num_filters_total, num_filters_total],
                initializer=tf.contrib.layers.xavier_initializer())
            self.transform_left = tf.matmul(self.h_pool_flat_q, W)
            self.sims = tf.reduce_sum(tf.mul(self.transform_left, self.h_pool_flat_t), 1, keep_dims=True)
            l2_loss += tf.nn.l2_loss(W)

        connect_len = 2 * num_filters_total + 1 + extra_features
        self.new_input = tf.concat(1, [self.h_pool_flat_q, self.sims, self.h_pool_flat_t, self.features], name='new_input')

        if architecture == 3 or architecture == 5:
            connect_len = 2 * num_filters_total + extra_features
            self.new_input = tf.concat(1, [self.h_pool_flat_q, self.h_pool_flat_t, self.features], name='new_input')

        # hidden layer
        with tf.name_scope("hidden"):
            W = tf.get_variable(
                "W_hidden",
                shape=[connect_len, num_neurons_fc],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_neurons_fc]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.hidden_output = tf.nn.elu(tf.nn.xw_plus_b(self.new_input, W, b, name="hidden_output"))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")

        # hidden layer 2
        with tf.name_scope("hidden_2"):
            W = tf.get_variable(
                "W_hidden_2",
                shape=[num_neurons_fc, num_neurons_fc_2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_neurons_fc_2]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.hidden_output_2 = tf.nn.elu(tf.nn.xw_plus_b(self.h_drop, W, b, name="hidden_output"))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden_output_2, self.dropout_keep_prob, name="hidden_output_drop")

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_neurons_fc_2, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")

            W_last = tf.get_variable(
                "W_last",
                shape=[2, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            b_last = tf.Variable(tf.constant(0.1, shape=[1]), name="b_last")

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            if architecture == 1:
                self.scores = tf.reduce_sum(tf.mul(self.h_pool_flat_q, self.h_pool_flat_t), axis=1, keep_dims=True)
            elif architecture == 2:
                self.scores = self.sims
            elif architecture == 5:
                two_scores = tf.concat(1, [self.sims, self.scores], name='two_scores')
                self.scores = tf.nn.xw_plus_b(two_scores, W_last, b_last, name="scores")

        with tf.device('/cpu:0'):
            with tf.name_scope("loss"):
                main_loss = calculate_loss(self.scores, self.pos_ind, self.neg_ind)

                self.loss = main_loss
