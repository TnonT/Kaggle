# -*- coding = utf-8 -*-

# @author:黑白
# @contact:1808132036@qq.com
# @time:19-2-26下午3:49
# @file:rnn.py

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper

class BiLSTM(object):
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
        :param label: with shape[batch_size]
        :return: feeddict
        """
        feed_dict = {
            self.sen: sen,
            self.truth: truth,
            self.sen_len: sen_len
        }
        return feed_dict

    def _BiLSTM(self, inputs, inputs_len, scope, reuse=False):
        """

        :param inputs: with shape[batch_size,sen_len, dim]
        :param inputs_len: with shape[batch_size]
        :param scope: variable scope
        :param reuse: weather reuse the variable
        :return: outputs with shape[batch_size, sen_len, 2*args.lstm.units]
        """7
        with tf.variable_scope(scope, reuse=reuse):
            cell = LSTMCell(self.args.lstm_units)
            drop_cell = lambda : DropoutWrapper(cell, output_keep_prob=self.args.dropout_rate)
            cell_fw, cell_bw = drop_cell(), drop_cell()
            batch_size = tf.shape(inputs)[0]
            init_state_fw = cell_fw.zero_state(batch_size, tf.float64)
            init_state_bw = cell_bw.zero_state(batch_size, tf.float64)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                         cell_bw=cell_bw,
                                                         inputs=inputs,
                                                         sequence_length=inputs_len,
                                                         initial_state_fw=init_state_fw,
                                                         initial_state_bw=init_state_bw,
                                                         dtype=tf.float64
                                                         )
            return tf.concat(outputs, axis=2)

    def _create_model_graph(self, embed_matrix, iter_num, l2_lambda=0.001):
        # ===============  Embedding Layer  =========================
        with tf.device('/cpu:0'):
            self.word_embeddings = tf.get_variable('embedding', trainable=True,
                                                   initializer=tf.constant(embed_matrix), dtype=tf.float64)
        sen_emb = tf.nn.embedding_lookup(self.word_embeddings, self.sen)

        # ===============  BiLSTM  ==================================
        sen = self._BiLSTM(sen_emb, self.sen_len, 'BiLSTM')[:, -1, :]

        # ===============  Forward Network  =========================
        with tf.variable_scope("feed_forward_network"):
            initializer = tf.random_normal_initializer(0.0, 0.1)

            with tf.variable_scope("feed_forward_layer1"):
                inputs = tf.nn.dropout(sen, self.args.dropout_rate)
                outputs = tf.layers.dense(inputs, 128, tf.nn.relu, kernel_initializer=initializer)
            with tf.variable_scope('feed_forward_layer2'):
                outputs = tf.nn.dropout(outputs, self.args.dropout_rate)
                self.logits = tf.layers.dense(outputs, 2, tf.nn.tanh, kernel_initializer=initializer)

        with tf.variable_scope("acc"):
            predict = tf.equal(tf.argmax(self.logits, axis=1), self.truth)
            self.accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

        with tf.variable_scope('loss'):
            # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.truth, logits=self.logits)
            # loss = tf.reduce_mean(losses, name='loss_val')
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.truth, logits=self.logits))
            weights = [v for v in tf.trainable_variables() if('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            self.loss = loss + l2_loss

        with tf.variable_scope("training"):
            # global step
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                self.args.learning_rate,   # base learning rate
                global_step,
                iter_num,
                self.args.learning_rate_decay
            )
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)




