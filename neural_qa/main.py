#! /usr/bin/env python
# coding=utf-8

import datetime
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from nltk.tokenize.toktok import ToktokTokenizer
from tensorflow.contrib import learn

import data_helpers
from text_cnn_char import TextCNNChar
from text_cnn_legacy import TextCNN_legacy
from text_rnn import TextRNN

tknzr = ToktokTokenizer()


def my_tokenizer(iterator):
    for value in iterator:
        value = value.replace('-', " - ")
        value = value.replace('/', " / ")
        value = value.lower()
        yield tknzr.tokenize(value)

def my_tokenizer_lambda(iterator):
    for value in iterator:
        value = value.replace('-', " - ")
        value = value.replace('/', " / ")
        value = value.lower()
        # NOTE THESE ARE FOR TOKENIZING LAMBDA CORRECTLY
        value = value.replace(".", " ")
        value = value.replace("_", " ")
        value = tknzr.tokenize(value)
        yield value

def string_from_ind(token_inds):
    tree_question = []
    for voc_ind in token_inds:
        if voc_ind == 0:
            break
        tree_question.append(inv_vocab[voc_ind])
    return u" ".join(tree_question)

# Log everything
tf.logging.set_verbosity(tf.logging.DEBUG)

# Parameters
# ==================================================

tf.flags.DEFINE_string("training_data_path", "./data/example.tsv", "Training source")
tf.flags.DEFINE_string("vocab_path", "./vocab", "Vocab of training + test")
tf.flags.DEFINE_string("glove", None, "Glove file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("testing_data_path", "./data/example.tsv", "Questions to test on")
tf.flags.DEFINE_string("validation_data_path", "./data/validation_example.txt", "Questions from the official validation split")

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .10, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings (default: None)")
# tf.flags.DEFINE_string("word2vec", "/users/tillhaug/cnn_thesis/data/word2vec.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_dim_char", 30, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,4,6,8", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("filter_sizes_char", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("filter_sizes_layer_two", "3,5,7,9", "Filter size of the second convolution layer")
tf.flags.DEFINE_integer("num_filters", 172, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_filters_char", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_filters_layer_two", 64, "Number of filters in the second convolution layer")
tf.flags.DEFINE_integer("num_neurons_fc", 500, "Number of Neurons in the fully connected layer")
tf.flags.DEFINE_integer("num_neurons_fc_2", 500, "Number of Neurons in the 2nd fully connected layer")
tf.flags.DEFINE_integer("num_neurons_fc_3", 600, "Number of Neurons in the 2nd fully connected layer")
tf.flags.DEFINE_integer("rnn_hidden_size", 350, "RNN Hidden Size")
tf.flags.DEFINE_integer("rnn_num_layers", 1, "RNN Hidden Size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("target_lambda", 1.0, "Used for weight of main loss, target replication loss is 1 - that (between 0 and 1)")
tf.flags.DEFINE_float("margin", 2.0, "Margin for Max Margin Loss")
tf.flags.DEFINE_float("learning_rate", 7 * 1e-4, "Learning Rate for our dear ADAM")
tf.flags.DEFINE_string("loss_function", "maxmargin", "Which loss function should be used 'crossentropy' or 'maxmargin'")
tf.flags.DEFINE_integer("architecture", 5, "1-5: different CNN architectures, 6: legacy CNN, 7: RNN")
tf.flags.DEFINE_integer("ratio_neg_pos", 1, "The ratio of negative samples per positive")
tf.flags.DEFINE_float("ratio_training_data", 1, "The ratio of negative samples per positive")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("is_training", False, "True for training, False for testing")


# Testing Options
# tf.flags.DEFINE_string("model_path", "/scratch/snx3000/tillhaug/runs/1485208587/checkpoints/model-100500", "all the models used to score")
tf.flags.DEFINE_string("model_path", "", "all the models used to score, comma separated")
tf.flags.DEFINE_string("testing_file", "evaluator_input.tsv", "where to store predictions")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
sys.stdout.flush()

# Data Preparation
# ==================================================

data_path = FLAGS.testing_data_path
if FLAGS.is_training:
    data_path = FLAGS.training_data_path


# Load vocab that was generated based on training + test data
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(FLAGS.vocab_path)

# Load (question, tree, label)
print("start initial load")
sys.stdout.flush()

if FLAGS.is_training:
    ids, x_q, x_t, y, features = data_helpers.load_training_data_efficient_with_add_features(data_path, vocab_processor)
else:
    ids, x_q, x_t, y, features, answers = data_helpers.load_training_data_efficient_with_answer(data_path, vocab_processor)

print("initial load done")
sys.stdout.flush()

id_grouped = OrderedDict()
for v, k in enumerate(ids):
    if k not in id_grouped:
        id_grouped[k] = []
    id_grouped[k].append(v)

distinct_ids = np.array(list(id_grouped.keys()))
x_grouped_indices = np.array(list(id_grouped.values()))

if FLAGS.is_training:
    remove_indices = []
    for v, k in enumerate(x_grouped_indices):
        if all((el == np.array([1, 0])).all() for el in y[x_grouped_indices[v]]):
            remove_indices.append(v)
    x_grouped_indices = np.delete(x_grouped_indices, remove_indices)
    distinct_ids = np.delete(distinct_ids, remove_indices)


char_vocabulary = set()
max_token_length = 0
for token in vocab_processor.vocabulary_._mapping:
    max_token_length = max(max_token_length, len(token))
    for char in token:
        char_vocabulary.add(char)

char_vocabulary = list('X') + sorted(list(char_vocabulary))
char_vocabulary = dict([(j,i) for (i,j) in enumerate(char_vocabulary)])
inv_vocab = {v: k for k, v in vocab_processor.vocabulary_._mapping.items()}


if not FLAGS.is_training:
    x_q = x_q[np.hstack(x_grouped_indices)]
    x_t = x_t[np.hstack(x_grouped_indices)]
    features = features[np.hstack(x_grouped_indices)]
    y = y[np.hstack(x_grouped_indices)]

# This param is determined by vocab_creator
max_document_length = x_q.shape[1]

if FLAGS.is_training:
    # Use the validation dataset
    validation_ids = data_helpers.validation_questions(FLAGS.validation_data_path)

    xsorted = np.argsort(distinct_ids)
    ypos = np.searchsorted(distinct_ids[xsorted], validation_ids)
    indices = xsorted[ypos]
    dev_indices = x_grouped_indices[indices]

    mask = np.ones(len(x_grouped_indices), np.bool)
    mask[indices] = 0
    train_indices = x_grouped_indices[mask]

    train_indices = train_indices[0:int(len(train_indices)*FLAGS.ratio_training_data)]

    # # Randomly shuffle data
    np.random.seed(123)
    # shuffle_indices_question = np.random.permutation(np.arange(len(x_grouped_indices)))
    # x_grouped_shuffled_indices = x_grouped_indices[shuffle_indices_question]
    # # x_grouped_shuffled_indices = x_grouped_indices
    #
    # # Split train/test set
    # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(x_grouped_shuffled_indices)))
    # train_indices, dev_indices = x_grouped_shuffled_indices[:dev_sample_index], x_grouped_shuffled_indices[
    #                                                                             dev_sample_index:]

    x_q_train, x_q_dev = x_q[np.hstack(train_indices)], x_q[np.hstack(dev_indices)]
    x_t_train, x_t_dev = x_t[np.hstack(train_indices)], x_t[np.hstack(dev_indices)]
    y_train, y_dev = y[np.hstack(train_indices)], y[np.hstack(dev_indices)]
    features_train, features_dev = features[np.hstack(train_indices)], features[np.hstack(dev_indices)]

    # x_grouped_indices = dev_indices
    # x_q = x_q_dev
    # x_t = x_t_dev
    # y = y_dev
    print("Questions to train {:d}".format(len(train_indices)))
    print("Questions to test {:d}".format(len(dev_indices)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
else:
    print("Questions to train: NONE")
    print("Questions to test {:d}".format(len(x_grouped_indices)))

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Seq length: {:d}".format(max_document_length))
sys.stdout.flush()

if FLAGS.is_training:
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            if FLAGS.architecture <= 5:
                neural_net = TextCNNChar(
                    sequence_length=max_document_length,
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    embedding_dim_char=FLAGS.embedding_dim_char,
                    filter_sizes=list(map(lambda x: int(x) if x != "inf" else max_document_length, FLAGS.filter_sizes.split(","))),
                    filter_sizes_char=list(map(lambda x: int(x) if x != "inf" else max_document_length, FLAGS.filter_sizes_char.split(","))),
                    num_filters=FLAGS.num_filters,
                    num_filters_char=FLAGS.num_filters_char,
                    num_neurons_fc=FLAGS.num_neurons_fc,
                    num_neurons_fc_2=FLAGS.num_neurons_fc_2,
                    margin=FLAGS.margin,
                    loss_function=FLAGS.loss_function,
                    target_lambda=FLAGS.target_lambda,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    architecture=FLAGS.architecture,
                    batch_size=FLAGS.batch_size,
                    char_vocab_size=len(char_vocabulary),
                    max_token_length=max_token_length
                )
            elif FLAGS.architecture == 6:
                neural_net = TextCNN_legacy(
                    sequence_length=max_document_length,
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    embedding_dim_char=FLAGS.embedding_dim_char,
                    filter_sizes=list(map(lambda x: int(x) if x != "inf" else max_document_length, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    num_filters_char=FLAGS.num_filters_char,
                    num_neurons_fc=FLAGS.num_neurons_fc,
                    num_neurons_fc_2=FLAGS.num_neurons_fc_2,
                    margin=FLAGS.margin,
                    num_filters_layer_two=FLAGS.num_filters_layer_two,
                    filter_sizes_layer_two=list(map(lambda x: int(x) if x != "inf" else max_document_length, FLAGS.filter_sizes_layer_two.split(","))),
                    char_vocab_size=len(char_vocabulary),
                    max_token_length=max_token_length,
                    filter_sizes_char=list(map(lambda x: int(x) if x != "inf" else max_document_length,
                                               FLAGS.filter_sizes_char.split(","))),

                )
            elif FLAGS.architecture == 7:
                neural_net = TextRNN(
                    sequence_length=max_document_length,
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    embedding_dim_char=FLAGS.embedding_dim_char,
                    num_filters_char=FLAGS.num_filters_char,
                    num_neurons_fc=FLAGS.num_neurons_fc,
                    num_neurons_fc_2=FLAGS.num_neurons_fc_2,
                    margin=FLAGS.margin,
                    rnn_num_layers=FLAGS.rnn_num_layers,
                    loss_function=FLAGS.loss_function,
                    rnn_hidden_size=FLAGS.rnn_hidden_size,
                    max_token_length=max_token_length,
                    char_vocab_size=len(char_vocabulary),
                    filter_sizes_char=list(map(lambda x: int(x) if x != "inf" else max_document_length,
                                               FLAGS.filter_sizes_char.split(","))),
                )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # optimizer = tf.train.AdagradOptimizer(1e-3)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(neural_net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            rand = os.urandom(10)
            timestamp_clock = rand.hex()
            timestamp += "_" + timestamp_clock
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", neural_net.loss)
            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=300)

            tf.add_to_collection('scores', neural_net.scores)
            tf.add_to_collection('input_x_q', neural_net.input_x_q)
            tf.add_to_collection('input_x_t', neural_net.input_x_t)
            tf.add_to_collection('input_x_q_char', neural_net.input_x_q_char)
            tf.add_to_collection('input_x_t_char', neural_net.input_x_t_char)
            tf.add_to_collection('input_y', neural_net.input_y)
            tf.add_to_collection('features', neural_net.features)
            tf.add_to_collection('input_x_q_len', neural_net.input_x_q_len)
            tf.add_to_collection('input_x_t_len', neural_net.input_x_t_len)
            tf.add_to_collection('dropout_keep_prob', neural_net.dropout_keep_prob)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            if FLAGS.word2vec:
                # initial matrix with random uniform
                initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec file {}\n".format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in range(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1)
                            if ch == b' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch.decode("ISO-8859-1"))
                        idx = vocab_processor.vocabulary_.get(word)
                        if idx != 0:
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)

                sess.run(neural_net.W.assign(initW))

            if FLAGS.glove:
                # initial matrix with random uniform
                initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load Glove file {}\n".format(FLAGS.glove))
                with open(FLAGS.glove, "r", encoding="UTF8") as f:
                    header = f.readline()
                    vocab_size = 400000
                    for line in range(vocab_size):
                        line = f.readline().rstrip()
                        head, *tail = line.split(" ")
                        idx = vocab_processor.vocabulary_.get(head)
                        if idx != 0:
                            initW[idx] = np.array(tail, dtype='float32')

                sess.run(neural_net.W.assign(initW))


            def train_step(x_q_batch, x_t_batch, y_batch, pos_ind, neg_ind, features_batch):
                """
                A single training step
                """
                x_q_batch_char = np.zeros([len(x_q_batch), max_document_length, max_token_length], dtype=np.int)
                for i, question in enumerate(x_q_batch):
                    for j, token_ind in enumerate(question):
                        if token_ind == 0:
                            break
                        token = inv_vocab[token_ind]
                        for k, char in enumerate(token):
                            x_q_batch_char[i][j][k] = char_vocabulary[char]

                x_t_batch_char = np.zeros([len(x_t_batch), max_document_length, max_token_length], dtype=np.int)
                for i, question in enumerate(x_t_batch):
                    for j, token_ind in enumerate(question):
                        if token_ind == 0:
                            break
                        token = inv_vocab[token_ind]
                        for k, char in enumerate(token):
                            x_t_batch_char[i][j][k] = char_vocabulary[char]


                feed_dict = {
                    neural_net.input_x_q: x_q_batch,
                    neural_net.input_x_t: x_t_batch,
                    neural_net.input_x_q_char: x_q_batch_char,
                    neural_net.input_x_t_char: x_t_batch_char,
                    neural_net.input_y: y_batch,
                    neural_net.pos_ind: pos_ind,
                    neural_net.neg_ind: neg_ind,
                    neural_net.features: features_batch,
                    neural_net.input_x_q_len: [len(np.where(x[::-1] != 0)[0]) for x in x_q_batch],
                    neural_net.input_x_t_len: [len(np.where(x[::-1] != 0)[0]) for x in x_t_batch],
                    neural_net.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss = sess.run(
                    [train_op, global_step, train_summary_op, neural_net.loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}".format(time_str, step, loss))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_q_batch_in, x_t_batch_in, y_batch_in, dev_indices_in, features_in, writer=None):
                """
                Evaluates model on a dev set
                """
                num_parts = len(x_q_batch_in) // FLAGS.batch_size

                x_q_part = np.array_split(x_q_batch_in, num_parts)
                x_t_part = np.array_split(x_t_batch_in, num_parts)
                y_part = np.array_split(y_batch_in, num_parts)
                features_part = np.array_split(features_in, num_parts)

                scores = None
                for i in range(0, len(y_part)):
                    x_q_batch_char = np.zeros([len(x_q_part[i]), max_document_length, max_token_length], dtype=np.int)
                    for j, question in enumerate(x_q_part[i]):
                        for k, token_ind in enumerate(question):
                            if token_ind == 0:
                                break
                            token = inv_vocab[token_ind]
                            for l, char in enumerate(token):
                                x_q_batch_char[j][k][l] = char_vocabulary[char]

                    x_t_batch_char = np.zeros([len(x_t_part[i]), max_document_length, max_token_length], dtype=np.int)
                    for j, question in enumerate(x_t_part[i]):
                        for k, token_ind in enumerate(question):
                            if token_ind == 0:
                                break
                            token = inv_vocab[token_ind]
                            for l, char in enumerate(token):
                                x_t_batch_char[j][k][l] = char_vocabulary[char]

                    feed_dict = {
                        neural_net.input_x_q: x_q_part[i],
                        neural_net.input_x_t: x_t_part[i],
                        neural_net.input_x_q_char: x_q_batch_char,
                        neural_net.input_x_t_char: x_t_batch_char,
                        neural_net.features: features_part[i],
                        neural_net.input_y: y_part[i],
                        neural_net.input_x_q_len: [len(np.where(x[::-1] != 0)[0]) for x in x_q_part[i]],
                        neural_net.input_x_t_len: [len(np.where(x[::-1] != 0)[0]) for x in x_t_part[i]],
                        neural_net.dropout_keep_prob: 1.0
                    }
                    step, _scores = sess.run([global_step, neural_net.scores], feed_dict)
                    if scores is None:
                        scores = _scores
                    else:
                        scores = np.vstack((scores, _scores))

                # Calculate Question Accuracy
                right = 0.0
                wrong = 0.0
                accumulator = 0
                for q_indices in dev_indices_in:
                    sub = []
                    for x, _ in enumerate(q_indices):
                        sub.append(x + accumulator)
                    s = scores[sub]
                    # Do this to avoid reporting false high scores due to stanford oracle
                    p = np.random.permutation(len(s))
                    s = s[p]
                    index = np.transpose(s)[0].argmax()
                    high_score = np.transpose(s)[0][index]
                    max_ties_index = np.argwhere(np.transpose(s)[0] == np.amax(np.transpose(s)[0]))
                    num_max_ties = len(max_ties_index)
                    res = y_batch_in[p[index] + accumulator]

                    quest = x_q[q_indices][p[index]]
                    actual_question = string_from_ind(quest)

                    tree = x_t[q_indices][p[index]]
                    tree_question = string_from_ind(tree)

                    if res[1] == 1:
                        right += 1
                        print(u"Right: {} -  {} - index {:g} - score {:g} - ties {:g}".format(actual_question.encode("UTF8"), tree_question.encode("UTF8"), index, high_score, num_max_ties))
                    else:
                        wrong += 1
                        print(u"Wrong: {} -  {} - index {:g} - score {:g} - ties {:g}".format(actual_question.encode("UTF8"), tree_question.encode("UTF8"), index, high_score, num_max_ties))

                    accumulator += len(q_indices)

                question_accuracy = right / (right + wrong)
                time_str = datetime.datetime.now().isoformat()
                print("{}: right: {:g}, wrong: {:g}, question acc {:g}".format(time_str, right, wrong,
                                                                               question_accuracy))

                return scores


            # # Generate batches
            # batches = data_helpers.batch_iter(list(zip(x_q_train, x_t_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            #
            # for batch in batches:
            #     x_q_batch, x_t_batch, y_batch = zip(*batch)
            #     train_step(x_q_batch, x_t_batch, y_batch)
            #     current_step = tf.train.global_step(sess, global_step)
            #     if current_step % FLAGS.evaluate_every == 0:
            #         print("\nEvaluation:")
            #         dev_step(x_q_dev, x_t_dev, y_dev, dev_indices, writer=dev_summary_writer)
            #         print("")
            #     if current_step % FLAGS.checkpoint_every == 0:
            #         path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #         print("Saved model checkpoint to {}\n".format(path))
            #


            # In each step take batch_size / 2 questions
            # from each one take one positive and one negative
            loop_index = 0
            total = 0

            num_diff_questions = FLAGS.batch_size // (FLAGS.ratio_neg_pos + 1)
            num_negatives = FLAGS.ratio_neg_pos
            while total < 140000:
                np.random.shuffle(train_indices)

                # Split train/test set
                indices_for_minibatch = train_indices[:num_diff_questions]
                pos_ind = []
                neg_ind = []
                for loop_train_indices in indices_for_minibatch:
                    np.random.shuffle(loop_train_indices)
                    y_batch = y[loop_train_indices]

                    positives = 0
                    negatives = 0
                    for i, yy in enumerate(y_batch):
                        if yy[0] == 0 and positives < 1:
                            pos_ind.append(loop_train_indices[i])
                            positives += 1
                        elif negatives < num_negatives:
                            neg_ind.append(loop_train_indices[i])
                            negatives += 1
                        if positives == 1 and negatives == num_negatives:
                            break

                bl = np.concatenate((pos_ind, neg_ind))
                indices = np.arange(len(bl))
                pos_ind, neg_ind = indices[:len(pos_ind)], indices[len(pos_ind):]
                train_step(x_q[bl], x_t[bl], y[bl], pos_ind, neg_ind, features[bl])

                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    scores = dev_step(x_q_dev, x_t_dev, y_dev, dev_indices, features_dev, writer=dev_summary_writer)
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

                    # Store scores for super duper ensemble
                    fname_ending = FLAGS.validation_data_path[-5:]
                    f = open('temp_steps/' + timestamp + "_" + str(current_step) + "_" + fname_ending, 'w')
                    scores.tofile(f)
                    f.close()

                total += 1
                loop_index += 1
                loop_index %= len(train_indices)

else:
    # Testing
    # ==================================================

    scores = None
    models_to_use = FLAGS.model_path.split(",")
    print("test with")
    for model_path in models_to_use:
       with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            def scores_for_model(x_q_batch_in, x_t_batch_in, y_batch_in, features_in):
                """
                Evaluates model on a dev set
                """
                num_parts = len(x_q_batch_in) // FLAGS.batch_size

                x_q_part = np.array_split(x_q_batch_in, num_parts)
                x_t_part = np.array_split(x_t_batch_in, num_parts)
                y_part = np.array_split(y_batch_in, num_parts)
                features_part = np.array_split(features_in, num_parts)

                scores = None
                for i in range(0, len(y_part)):
                    x_q_batch_char = np.zeros([len(x_q_part[i]), max_document_length, max_token_length],
                                              dtype=np.int)
                    for j, question in enumerate(x_q_part[i]):
                        for k, token_ind in enumerate(question):
                            if token_ind == 0:
                                break
                            token = inv_vocab[token_ind]
                            for l, char in enumerate(token):
                                x_q_batch_char[j][k][l] = char_vocabulary[char]

                    x_t_batch_char = np.zeros([len(x_t_part[i]), max_document_length, max_token_length],
                                              dtype=np.int)
                    for j, question in enumerate(x_t_part[i]):
                        for k, token_ind in enumerate(question):
                            if token_ind == 0:
                                break
                            token = inv_vocab[token_ind]
                            for l, char in enumerate(token):
                                x_t_batch_char[j][k][l] = char_vocabulary[char]

                    feed_dict = {
                        tf.get_collection('input_x_q')[0]: x_q_part[i],
                        tf.get_collection('input_x_t')[0]: x_t_part[i],
                        tf.get_collection('features')[0]: features_part[i],
                        tf.get_collection('input_x_q_char')[0]: x_q_batch_char,
                        tf.get_collection('input_x_t_char')[0]: x_t_batch_char,
                        tf.get_collection('input_y')[0]: y_part[i],
                        tf.get_collection('input_x_q_len')[0]: [len(np.where(x[::-1] != 0)[0]) for x in
                                                                x_q_part[i]],
                        tf.get_collection('input_x_t_len')[0]: [len(np.where(x[::-1] != 0)[0]) for x in
                                                                x_t_part[i]],
                        tf.get_collection('dropout_keep_prob')[0]: 1.0
                    }
                    _scores = sess.run(tf.get_collection('scores')[0], feed_dict)
                    if scores is None:
                        scores = _scores
                    else:
                        scores = np.vstack((scores, _scores))


                print(scores.shape)
                rows, cols = scores.shape
                # This is to support legacy models with two columns of scores
                if cols > 1:
                    scores = scores[:, 0:1]

                print(scores.shape)
                sys.stdout.flush()

                # Bring score to range 0 to 1
                min_score = np.amin(scores)
                max_score = np.amax(scores)
                scores = (scores - min_score) / (max_score - min_score)

                return scores


            def test(scores, dev_indices_in, y_batch_in):
                # Calculate Question Accuracy
                right = 0.0
                wrong = 0.0
                accumulator = 0
                official_validator = []
                for q_indices in dev_indices_in:
                    sub = []
                    for x, _ in enumerate(q_indices):
                        sub.append(x + accumulator)
                    s = scores[sub]
                    # Do this to avoid reporting false high scores due to stanford oracle
                    p = np.random.permutation(len(s))
                    s = s[p]

                    scores_for_np = np.transpose(s)[0]
                    index = scores_for_np.argmax()
                    high_score = scores_for_np[index]
                    max_ties_index = np.argwhere(scores_for_np == np.amax(scores_for_np))
                    num_max_ties = len(max_ties_index)
                    res = y_batch_in[p[index] + accumulator]

                    quest = x_q[q_indices][p[index]]
                    actual_question = string_from_ind(quest)

                    tree = x_t[q_indices][p[index]]
                    tree_question = string_from_ind(tree)

                    q_id = ids[q_indices][p[index]]
                    q_answer = answers[q_indices][p[index]]
                    official_validator.append((q_id, q_answer))

                    ind_scores = np.argsort(scores_for_np)[::-1]

                    if res[1] == 1:
                        right += 1
                        print(u"{} Right: {} -  {} - index {:g} - score {:g} - ties {:g}".format(q_id.encode("UTF8"),actual_question.encode("UTF8"), tree_question.encode("UTF8"), p[index], high_score, num_max_ties))
                    else:
                        wrong += 1
                        print(u"{} Wrong: {} -  {} - index {:g} - score {:g} - ties {:g}".format(q_id.encode("UTF8"),actual_question.encode("UTF8"), tree_question.encode("UTF8"), p[index], high_score, num_max_ties))
                        for i, ind in enumerate(ind_scores):
                            tree = x_t[q_indices][p[ind]]
                            tree_question = string_from_ind(tree)
                            if i > 10:
                                break
                            print(u"   {:g}   {:g}   {}".format((y_batch_in[p[ind] + accumulator])[1], scores_for_np[ind], tree_question.encode("UTF8")))

                    accumulator += len(q_indices)

                question_accuracy = right / (right + wrong)
                time_str = datetime.datetime.now().isoformat()
                print("{}: right: {:g}, wrong: {:g}, question acc {:g}".format(time_str, right, wrong, question_accuracy))
                data_helpers.answer_to_official(official_validator, FLAGS.testing_file)


            print(model_path)
            sys.stdout.flush()
            saver = tf.train.import_meta_graph(model_path + ".meta")
            saver.restore(sess, model_path)
            new_scores = scores_for_model(x_q, x_t, y, features)

            if scores is not None:
                scores += new_scores
            else:
                scores = new_scores

    test(scores, x_grouped_indices, y)