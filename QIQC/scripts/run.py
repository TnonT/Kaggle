# -*- coding = utf-8 -*-

# @author:黑白
# @contact:1808132036@qq.com
# @time:19-2-26下午3:52
# @file:run.py
from time import time
import numpy as np
import argparse
import tensorflow as tf
from tqdm import tqdm
from model.bilstm import BiLSTM
from model.rnn import RNN
from utils.data_processing import read_data, load_embedding, make_embedding_matrixs, get_train_batch


train_x, train_y, dev_x, dev_y, test_x, train_x_len, dev_x_len, test_x_len, word_index = read_data()

def train(sess, args, model, epoch, batchs_per_epoch):

    ptbr = tqdm(range(batchs_per_epoch))

    total_loss = []
    total_acc = []

    for batch_num in ptbr:
        ptbr.set_description("Epoch {}".format(epoch+1))
        train_batch_x, train_batch_y, train_batch_x_len = get_train_batch(train_x, train_y, train_x_len, args.batch_size, batch_num)
        feed_dict = model._create_feed_dict(train_batch_x, train_batch_y, train_batch_x_len)
        _, loss, acc = sess.run([model.train_op, model.loss, model.accuracy], feed_dict=feed_dict)
        total_loss.append(loss)
        total_acc.append(acc)
    # return total_loss/batchs_per_epoch, total_acc/batchs_per_epoch
    return np.mean(total_loss), np.mean(total_acc)

def main(args):
    embedding_index = load_embedding(args.embedding_file)
    embedding_matrix = make_embedding_matrixs(word_index, embedding_index)
    batchs_per_epoch = len(train_x_len) // args.batch_size + 1
    # 定义初始化函数 tensorflow v1.8
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    print("初始化训练模型...")
    # train model
    train_start_t = time()
    with tf.name_scope("Training"):
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE, initializer=initializer):
            train_model = BiLSTM(args, embedding_matrix, batchs_per_epoch)
            # train_model = RNN(args, embedding_matrix, batchs_per_epoch)
    print(time() - train_start_t)
    print("训练模型初始化结束...")
    print("初始化验证模型...")
    dev_start_t = time()
    # dev model
    with tf.name_scope("Dev"):
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE, initializer=initializer):
            dev_model = BiLSTM(args, embedding_matrix, batchs_per_epoch)
            # dev_model = RNN(args, embedding_matrix, batchs_per_epoch)
    print(time() - dev_start_t)
    print("验证初始化结束...")

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()

        for epoch in range(args.num_epoch):
            train_loss, train_acc = train(sess, args, train_model, epoch, batchs_per_epoch)
            print('train loss: {0}, train acc: {1}'.format(train_loss, train_acc))

            feed_dict = dev_model._create_feed_dict(dev_x[:2000], dev_y[:2000], dev_x_len[:2000])
            dev_loss, dev_acc = sess.run([dev_model.loss, dev_model.accuracy], feed_dict=feed_dict)
            print('dev loss: {0}, dev acc: {1}'.format(dev_loss, dev_acc))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding_file', type=str, default='../embeddings/glove.840B.300d/glove.840B.300d.txt', help='embedding file')
    parser.add_argument('--lstm_units', type=int, default=128, help='lstm units')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_epoch', type=int, default=10, help='epoch num')
    parser.add_argument('--dropout_rate', type=float, default=0.25, help='dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help="learning rate")
    parser.add_argument('--learning_rate_decay', type=float, default=0.8, help="learning rate decay")
    args = parser.parse_args()

    # run
    main(args)