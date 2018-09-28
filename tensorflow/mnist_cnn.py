import numpy as np
import pandas as pd
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data

def conv2d3x3(input, filter, bias):
  conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
  return conv + bias

def maxpool2x2(input):
  return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def train_data():
  # mnist = tf.keras.datasets.mnist
  mnist = input_data.read_data_sets("data/", one_hot=True)
  
  w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
  b1 = tf.Variable(tf.constant(.1, shape=[32]))

  w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 32], stddev=0.1))
  b2 = tf.Variable(tf.constant(.1, shape=[32]))

  x_input = tf.placeholder(tf.float32, [None, 28, 28, 1])
  y_input = tf.placeholder(tf.float32, [None, 10])

  cnn_layer1 = maxpool2x2(tf.nn.relu(conv2d3x3(x_input, w1, b1)))
  cnn_layer2 = maxpool2x2(tf.nn.relu(conv2d3x3(cnn_layer1, w2, b2)))

  cnn2 = tf.reshape(cnn_layer2, [-1, 7 * 7 * 32]) 

  fw1 = tf.Variable(tf.truncated_normal([7 * 7 * 32, 32], stddev=0.1))
  fb1 = tf.Variable(tf.constant(.1, shape=[32]))
  fc1 = tf.nn.relu(tf.matmul(cnn2, fw1) + fb1)
  fc1 = tf.nn.dropout(fc1, 0.7)

  fw2 = tf.Variable(tf.truncated_normal([32, 10], stddev=.1))
  fb2 = tf.Variable(tf.constant(.1, shape=[10]))
  fc2 = tf.matmul(fc1, fw2) + fb2

  loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=fc2)
  opt = tf.train.AdamOptimizer().minimize(loss)

  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc2, axis=1), tf.argmax(y_input, axis=1)), 'float'))

  # Batch_Size = 128
  # print('batch count:', x_train.shape[0])
  # Batch_Count = math.ceil(x_train.shape[0] / Batch_Size)

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # batch = mnist.train.next_batch(100)
    # batchInput = batch[0].reshape([-1, 28, 28, 1])
    # batchLabels = batch[1]
    # print('layer1:', session.run(tf.shape(cnn_layer1), feed_dict={x_input: batchInput, y_input: batchLabels}))
    # print('layer1:', session.run(tf.shape(cnn_layer2), feed_dict={x_input: batchInput, y_input: batchLabels}))
    for di in range(1000):
      batch = mnist.train.next_batch(100)
      batchInput = batch[0].reshape([-1, 28, 28, 1])
      batchLabels = batch[1]
      session.run(opt, feed_dict={x_input: batchInput, y_input: batchLabels})

      if di % 100 is 0:
        print('The %d epoch, Accuracy is %f' % (di, session.run(accuracy, feed_dict={x_input: batchInput, y_input: batchLabels})))