import numpy as np
import pandas as pd
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data

def train_data():
  # mnist = tf.keras.datasets.mnist
  mnist = input_data.read_data_sets("data/", one_hot=True)

  ValueCount = 10
  HiddenLayerCount = 20
  # (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # x_train = x_train.reshape((-1, 28 * 28))
  # y_train = pd.get_dummies(y_train)[list(range(10))]
  # x_test = x_test.reshape((-1, 28 * 28))
  # y_test = pd.get_dummies(y_test)[list(range(10))]

  w1 = tf.Variable(tf.truncated_normal([28 * 28, HiddenLayerCount], stddev=0.1))
  b1 = tf.Variable(tf.zeros([HiddenLayerCount]))

  w2 = tf.Variable(tf.truncated_normal([HiddenLayerCount, ValueCount], stddev=0.1))
  b2 = tf.Variable(tf.zeros([ValueCount]))

  x_input = tf.placeholder(tf.float32, [None, 28 * 28])
  y_input = tf.placeholder(tf.float32, [None, ValueCount])

  hidden_layer = tf.matmul(x_input, w1) + b1
  hidden_layer = tf.nn.relu(hidden_layer)

  output = tf.matmul(hidden_layer, w2) + b2
  output = tf.nn.relu(output)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_input))
  
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=.1)
  operation = optimizer.minimize(loss)

  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y_input, axis=1)), 'float'))

  # Batch_Size = 128
  # print('batch count:', x_train.shape[0])
  # Batch_Count = math.ceil(x_train.shape[0] / Batch_Size)

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for di in range(10000):
      batch = mnist.train.next_batch(100)
      batchInput = batch[0]
      batchLabels = batch[1]
      session.run(operation, feed_dict={x_input: batchInput, y_input: batchLabels})

      if di % 1000 is 0:
        print('The %d epoch, Accuracy is %f' % (di, session.run(accuracy, feed_dict={x_input: batchInput, y_input: batchLabels})))