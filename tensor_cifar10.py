# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.datasets.cifar10 import load_data
import numpy as np


def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


tf.set_random_seed(777)
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

(x_train, y_train), (x_test, y_test) = load_data()
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, depth=10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

learning_rate = 0.001
training_epochs = 30
batch_size = 100

keep_prob = tf.placeholder(tf.float32)


class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [None, 32, 32, 3])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            conv1 = tf.layers.conv2d(inputs=self.X, filters=32, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)
            flat = tf.reshape(dropout1, [-1, 32 * 16 * 16])
            dense1 = tf.layers.dense(inputs=flat, units=615, activation=tf.nn.relu)
            dropout2 = tf.layers.dropout(inputs=dense1, rate=0.5, training=self.training)

            self.logits = tf.layers.dense(inputs=dropout2, units=10)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})





with tf.Session() as sess:

    m1 = Model(sess, 'm1')
    sess.run(tf.global_variables_initializer())

    sess.run(tf.global_variables_initializer())

    print("Learning Started!")

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(50000 / batch_size)

        for i in range(total_batch):
            batch = next_batch(128, x_train, y_train_one_hot.eval())
            c, _ = m1.train(batch[0], batch[1])
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print("Learning Finished!")

    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
        test_accuracy += m1.get_accuracy(batch[0], batch[1])
    test_accuracy = test_accuracy / 10
    print("Accuracy: ", test_accuracy)


