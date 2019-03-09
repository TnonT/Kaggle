# -*- coding = utf-8 -*-

# @author:黑白
# @contact:1808132036@qq.com
# @time:19-2-28下午9:57
# @file:rnn.py
import tensorflow as tf
import numpy as np

class RNN(object):
    def __init__(self, args, embed_matrix, iter_num, is_trainging=True):
        self.args = args
        self._create_placeholder()
        self._create_model_graph(embed_matrix, iter_num)

    def _create_placeholder(self):
        self.sen = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.sen_len = tf.placeholder(dtype=tf.int32, shape=[None])
        self.truth = tf.placeholder(dtype=tf.int64, shape=[None])

    def _create_feed_dict(self, sen, truth, sen_len):
        """

        :param sen: with shape[batch_size, max_sen_len]
        :param truth: with shape[batch_size]
        :param sen_len: with shape[batch_size]
        :return: feed_dict
        """
        feed_dict = {
            self.sen: sen,
            self.truth: truth,
            self.sen_len: sen_len

        }
        return feed_dict


    def _create_model_graph(self, embed_matrix, iter_num, l2_lambda=0.001):
        # ===============  Embedding Layer  =========================
        with tf.device('/cpu:0'):
            self.word_embeddings = tf.get_variable('embedding', trainable=True,
                                                   initializer=tf.constant(embed_matrix), dtype=tf.float64)
        sen_emb = tf.nn.embedding_lookup(self.word_embeddings, self.sen)

        # ==============  rnn  ==============================
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.args.lstm_units, state_is_tuple=True)
        sen, _ = tf.nn.dynamic_rnn(cell=cell,
                                   dtype=tf.float64,
                                   sequence_length=self.sen_len,
                                   inputs=sen_emb)
        sen = sen[:, -1, :]

        # ==============   forward network  =====================
        with tf.variable_scope("feed_forward_network"):
            initializer = tf.random_normal_initializer(0.0, 0.1)
            with tf.variable_scope("feed_forward_layer1"):
                inputs = tf.nn.dropout(sen, self.args.dropout_rate)
                outputs = tf.layers.dense(inputs, 128, tf.nn.relu, kernel_initializer=initializer)
            with tf.variable_scope('feed_forward_layer2'):
                outputs = tf.nn.dropout(outputs, self.args.dropout_rate)
                results = tf.layers.dense(outputs, 2, tf.nn.tanh, kernel_initializer=initializer)
            self.logits = results

        with tf.variable_scope("acc"):
            self.predict = tf.equal(tf.argmax(self.logits, axis=1), self.truth)
            self.accuracy = tf.reduce_mean(tf.cast(self.predict, tf.float32))

        with tf.variable_scope('loss'):
            # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.truth, logits=self.logits)
            # loss = tf.reduce_mean(losses, name='loss_val')
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=self.logits))
            weights = [v for v in tf.trainable_variables() if('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            self.loss = loss + l2_loss

        # with tf.variable_scope("loss"):
        #     # self.label = tf.one_hot(self.label, depth=2, axis=1)
        #     # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.logits))
        #     # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=self.logits))
        #     self.truth_1 = tf.cast(self.truth, dtype=tf.int64)
        #     self.label = tf.one_hot(self.truth_1, depth=2, dtype=tf.float64)
        #     self.logits_1 = tf.cast(self.logits, dtype=tf.int64)
        #     self.logits_1 = tf.one_hot(self.logits_1, depth=2, dtype=tf.float64)
        #     # self.p = tf.nn.softmax(self.logits, axis=1)
        #     # self.p = tf.log(self.p)
        #     self.p = tf.log(self.logits_1)
        #     print(self.p)
        #     print(self.label)
        #     loss = -tf.reduce_mean(tf.reduce_sum(self.label * self.p, axis=1))
        #     weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
        #     l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
        #     loss += l2_loss
        #     self.loss = loss

        with tf.variable_scope("training"):
            # global step
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                self.args.learning_rate,  # base learning rate
                global_step,
                iter_num,
                self.args.learning_rate_decay
            )
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)

