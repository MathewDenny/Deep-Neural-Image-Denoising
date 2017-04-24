
"""
Created on Apr 22, 2017

@author: denny
"""

import tensorflow as tf
import numpy as np
import cv2
import time

def get_batch(X, Xn, size):
    a = np.random.choice(len(X), size, replace=False)

    return X[a], Xn[a]

class Denoiser:

    def __init__(self, input_dim, hidden_dim, epoch=10000, batch_size=50, learning_rate=0.001):
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        self.x_noised = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_noised')
        with tf.name_scope('encode'):
            self.weights1 = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32), name='weights')
            self.biases1 = tf.Variable(tf.zeros([hidden_dim]), name='biases')
            self.encoded = tf.nn.sigmoid(tf.matmul(self.x_noised, self.weights1) + self.biases1, name='encoded')
        with tf.name_scope('decode'):
            weights = tf.Variable(tf.random_normal([hidden_dim, input_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([input_dim]), name='biases')
            self.decoded = tf.matmul(self.encoded, weights) + biases
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded))))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def add_noise(self, data, mean, sigma):
        noise_type = 'gaussian'
        noisy_patches = []
        i = 1
        if noise_type == 'gaussian':
            for patch in data:
                print "On ",i,"/",len(data),"\r",
                i += 1
                n = np.random.normal(mean, sigma, np.shape(patch))
                added = patch + n
                cv2.normalize(added, added, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32FC3 )    
#                 noisy_patches.append(added)
                if len(noisy_patches) > 0:     
                    noisy_patches = np.vstack((noisy_patches, added))    
                else:
                    noisy_patches = added
            
            noisy_patches = np.matrix(noisy_patches)  
            return noisy_patches

    def train(self, data):
        print "Adding Noise!"
        data_noised = self.add_noise(data, 0, 10)
        with open('log.csv', 'w') as writer:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print "Training to denoise"
                for i in range(self.epoch):
                    for j in range(50):
                        batch_data, batch_data_noised = get_batch(data, data_noised, self.batch_size)
                        l, _ = sess.run([self.loss, self.train_op], feed_dict={self.x: batch_data, self.x_noised: batch_data_noised})
                    if i % 10 == 0:
                        print('epoch {0}: loss = {1}'.format(i, l))
                        self.saver.save(sess, './model.ckpt')
                        epoch_time = int(time.time())
                        row_str = str(epoch_time) + ',' + str(i) + ',' + str(l) + '\n'
                        writer.write(row_str)
                        writer.flush()
                self.saver.save(sess, './model.ckpt')

    def test(self, data):
        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')
            hidden, reconstructed = sess.run([self.encoded, self.decoded], feed_dict={self.x: data})
        print('input', data)
        print('compressed', hidden)
        print('reconstructed', reconstructed)
        return reconstructed

    def get_params(self):
        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')
            weights, biases = sess.run([self.weights1, self.biases1])
        return weights, biases