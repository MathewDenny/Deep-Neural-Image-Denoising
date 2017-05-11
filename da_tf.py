
"""
Created on Apr 22, 2017

@author: denny
"""

import tensorflow as tf
import numpy as np
import cv2
import time
import random
from collections import OrderedDict


from layer import AELayer
from numpy.core.multiarray import dtype

def get_batch(X, Xn, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a], Xn[a]

class Denoiser:

    def __init__(self, input_dim, hidden_dim, epoch=1500, batch_size=10, learning_rate=0.001):
        hidden_dim2 = 5 * hidden_dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        self.x_noised = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_noised')
        
        self.sparsity_level= np.repeat([0.05], hidden_dim).astype(np.float32) 
        #TODO check thhis
        self.sparse_reg = 0.001       
        
            
        with tf.name_scope('encode'):
#             with tf.Session() as sess:
#                 self.saver.restore(sess, './model_da1.ckpt')
#                 self.weights1, self.biases1 = sess.run([self.weights1, self.biases1])

#             self.weights1 = tf.Variable(self.xavier_init(input_dim, hidden_dim, 4), name='weights')
                #LAYER1
            self.weights1 = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32), name='weights')
            self.biases1 = tf.Variable(tf.zeros([hidden_dim]), name='biases')
            self.weights2 = tf.Variable(tf.random_normal([hidden_dim, hidden_dim2], dtype=tf.float32), name='weights2')
            self.biases2 = tf.Variable(tf.zeros([hidden_dim2]), name='biases2')
            
#             self.trained_weights1 = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32), name='trained_weights')
#             self.trained_biases1 = tf.Variable(tf.zeros([hidden_dim]), name='trained_biases')
#             self.trained_weights2 = tf.Variable(tf.random_normal([hidden_dim, hidden_dim2], dtype=tf.float32), name='trained_weights2')
#             self.trained_biases2 = tf.Variable(tf.zeros([hidden_dim2]), name='trained_biases2')
            
            self.encoded1 = tf.nn.sigmoid(tf.matmul(self.x_noised, self.weights1) + self.biases1, name='encoded1')
            self.encoded_pure = tf.matmul(self.x, self.weights1) + self.biases1
            self.encoded2 = tf.nn.sigmoid(tf.matmul(self.encoded1, self.weights2) + self.biases2, name='encoded2')
            
            self.encoded_out_1 = tf.nn.sigmoid((tf.matmul(self.x_noised, self.weights1) + self.biases1), name='encoded_out1')
            self.encoded_out_2 = tf.nn.sigmoid((tf.matmul(self.encoded_out_1, self.weights2) + self.biases2), name='encoded_out2')


#             tfMean = tf.reduce_mean(self.encoded)
#             
#             self.kl_div = self.kl_divergence(self.sparsity_level, tfMean)
        with tf.name_scope('decode'):
#             with tf.Session() as sess:
#                 self.saver.restore(sess, './model_da1.ckpt')
#                 self.d_weights2, self.d_biases2 = sess.run([self.d_weights2, self.d_biases2])
#             weights = tf.Variable(self.xavier_init(hidden_dim, input_dim, 4), name='weights')
            self.d_weights1 = tf.Variable(tf.random_normal([hidden_dim2, hidden_dim], dtype=tf.float32), name='d_weights')
            self.d_biases1 = tf.Variable(tf.zeros([hidden_dim]), name='biases')
            self.d_weights2 = tf.Variable(tf.random_normal([hidden_dim, input_dim], dtype=tf.float32), name='d_weights2')
            self.d_biases2 = tf.Variable(tf.zeros([input_dim]), name='biases2')
            
#             self.trained_d_weights1 = tf.Variable(tf.random_normal([hidden_dim2, hidden_dim], dtype=tf.float32), name='trained_d_weights')
#             self.trained_d_biases1 = tf.Variable(tf.zeros([hidden_dim]), name='trained_d_biases')
#             self.trained_d_weights2 = tf.Variable(tf.random_normal([hidden_dim, input_dim], dtype=tf.float32), name='trained_d_weights2')
#             self.trained_d_biases2 = tf.Variable(tf.zeros([input_dim]), name='trained_d_biases2')
            
            self.decoded1 = tf.nn.sigmoid(tf.matmul(self.encoded2, self.d_weights1) + self.d_biases1, name='decoded1')
            #op of final decode stage
            self.decoded2 = tf.nn.sigmoid(tf.matmul(self.encoded1, self.d_weights2) + self.d_biases2, name='decoded2')
            
            self.decoded_out_1 = tf.nn.sigmoid((tf.matmul(self.encoded_out_2, self.d_weights1) + self.d_biases1), name='decoded_out_1')
            self.decoded_out_2 = tf.nn.sigmoid((tf.matmul(self.decoded_out_1, self.d_weights2) + self.d_biases2), name='decoded_out_2')
            
        #self.loss = tf.reduce_sum((self.x - self.decoded) ** 2) + self.sparse_reg * self.kl_div
        #ootput of last stage
        self.loss1 = tf.sqrt(tf.reduce_mean(tf.square(self.decoded2 - self.x))) #+ self.sparse_reg * self.kl_div
        #output of inner stage
        self.loss2 = tf.sqrt(tf.reduce_mean(tf.square(self.encoded_pure - self.decoded1))) #+ self.sparse_reg * self.kl_div
        #loss for the deep network
        self.loss3 = tf.sqrt(tf.reduce_mean(tf.square(self.decoded_out_2 - self.x))) #+ self.sparse_reg * self.kl_div
        #self.train_op =  tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
        #self.train_op =  tf.contrib.opt.ScipyOptimizerInterface(self.loss, options={'maxiter': 100})
        # last stage
        self.train_op1 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss1)
        self.train_op2 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss2)
        self.train_op3 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss3)
        
        self.saver = tf.train.Saver()


            
    def xavier_init(self, fan_in, fan_out, const=4):

        low = -const * np.sqrt(6.0 / (fan_in + fan_out))
        high = const * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)

    def kl_divergence(self, p, p_hat):
        return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)
    
#     def Forward(self, x_in):
#         lin_h = tf.matmul(self.hiddenLayer.W, x_in) + self.hiddenLayer.b
#         h = self.activation(lin_h)
#         #notice sparsity_level read from numpy can be dtype64, should using astype to float32
#         kl_div = self.kl_divergence(self.sparsity_level, h)
# 
#         lin_output = tf.matmul(tf.transpose(self.hiddenLayer.W), h) + self.hiddenLayer.b_prime
#         return lin_output, kl_div
#     
#     def TrainNN(self):
#         #sen_vec = T.vector()
#         sen_vec= tf.placeholder(tf.float64, [300,1])
# 
#         updates = OrderedDict({})
#         output1, output2 = self.Forward(sen_vec)
#         cost = tf.reduce_sum((sen_vec - output1) ** 2)+ self.sparse_reg * output2
# 
#         self.gparams=tf.gradients(cost,self.params)
# 
#         for param, gparam in zip(self.params, self.gparams):
#             #print param
#             updates[self.batch_grad[param]] = self.batch_grad[param] + gparam
#         feed_dict = sen_vec
#         return cost, feed_dict
    
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
                cv2.normalize(added, added, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1 )    
#                 noisy_patches.append(added)
                if len(noisy_patches) > 0:     
                    noisy_patches = np.vstack((noisy_patches, added))    
                else:
                    noisy_patches = added
            
            noisy_patches = np.matrix(noisy_patches, dtype = np.float32)  
            return noisy_patches

    def train(self, data):
        print "Adding Noise!"
        data_noised = self.add_noise(data, 0, 0.01)
        with open('log.csv', 'w') as writer:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print "Training layer 1"
                for i in range(self.epoch):
                    for j in range(50):
                        batch_data, batch_data_noised = get_batch(data, data_noised, self.batch_size)
                        l, _ = sess.run([self.loss1, self.train_op1], feed_dict={self.x: batch_data, self.x_noised: batch_data_noised})
                    if i % 10 == 0:
                        print('epoch {0}: loss = {1}'.format(i, l))
                        v = sess.run(self.d_weights2)
                        print('self.d_weights2', v)

                        self.saver.save(sess, './model.ckpt')
                        epoch_time = int(time.time())
                        row_str = str(epoch_time) + ',' + str(i) + ',' + str(l) + '\n'
                        writer.write(row_str)
                        writer.flush()
                #self.saver.save(sess, './model_da1.ckpt')
                print "Training layer 2 (inner)"
                for i in range(self.epoch):
                    for j in range(50):
                        batch_data, batch_data_noised = get_batch(data, data_noised, self.batch_size)
                        l, _ = sess.run([self.loss2, self.train_op2], feed_dict={self.x: batch_data, self.x_noised: batch_data_noised})
                    if i % 10 == 0:
                        print('epoch {0}: loss = {1}'.format(i, l))
                        v = sess.run(self.d_weights1)
                        print('self.d_weights1', v)

                        self.saver.save(sess, './model.ckpt')
                        epoch_time = int(time.time())
                        row_str = str(epoch_time) + ',' + str(i) + ',' + str(l) + '\n'
                        writer.write(row_str)
                        writer.flush()
                print "Training Deep network"
                for i in range(self.epoch):
                    for j in range(50):
                        batch_data, batch_data_noised = get_batch(data, data_noised, self.batch_size)
                        l, _ = sess.run([self.loss3, self.train_op3], feed_dict={self.x: batch_data, self.x_noised: batch_data_noised})
                    if i % 10 == 0:
                        print('epoch {0}: loss = {1}'.format(i, l))
                        v = sess.run(self.d_weights1)
                        print('self.d_weights1', v)

                        self.saver.save(sess, './model.ckpt')
                        epoch_time = int(time.time())
                        row_str = str(epoch_time) + ',' + str(i) + ',' + str(l) + '\n'
                        writer.write(row_str)
                        writer.flush()
                self.saver.save(sess, './model_da2.ckpt')

    def test(self, data):
        with tf.Session() as sess:
            self.saver.restore(sess, './model_da2.ckpt')
            hidden, hidden2, hidden3, reconstructed = sess.run([self.encoded_out_1, self.encoded_out_2, self.decoded_out_1, self.decoded_out_2], feed_dict={self.x_noised: data})
        print('input', data)
        print('compressed', hidden)
        print('reconstructed', reconstructed)
        return reconstructed

    def get_params(self):
        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')
            weights, biases = sess.run([self.weights1, self.biases1])
        return weights, biases