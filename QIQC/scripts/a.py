# -*- coding = utf-8 -*-

# @author:黑白
# @contact:1808132036@qq.com
# @time:19-3-1下午2:20
# @file:a.py

import tensorflow as tf
import numpy as np

# a = tf.constant([2, 3], dtype=tf.int64)
# b = tf.constant([2, 6], dtype=tf.int64)
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     # if tf.equal(a,b):
#     #     print("Equal")
#     # else:
#     #     print("Not Equal")
#     predict = tf.equal(a,b)
#     print(predict.eval())

a = np.ones(8).reshape([2, 2, 2])
tensor_a = tf.constant(a)
tensor_b = tensor_a[:, -1, :]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(tf.shape(tensor_b))
    print(tensor_b)
    print(tensor_b.eval())