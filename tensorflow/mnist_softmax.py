import numpy as np
import pandas as pd
import tensorflow as tf
import math

def train_data():
  mnist = tf.keras.datasets.mnist
  ValueCount = 10
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.reshape((-1, 28 * 28))
  y_train = pd.get_dummies(y_train)[list(range(10))]
  x_test = x_test.reshape((-1, 28 * 28))
  y_test = pd.get_dummies(y_test)[list(range(10))]

  w = tf.Variable(tf.random_normal([28 * 28, ValueCount]))
  b = tf.Variable(tf.zeros([ValueCount]))
  x_input = tf.placeholder(tf.float32, [None, 28 * 28])
  y_input = tf.placeholder(tf.float32, [None, ValueCount])

  y_pred = tf.nn.softmax(tf.matmul(x_input, w) + b)
  loss = tf.reduce_mean(tf.square(y_pred - y_input))

  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y_input, axis=1)), 'float'))
  optimizer = tf.train.AdamOptimizer()
  operation = optimizer.minimize(loss)

  Batch_Size = 16
  Batch_Count = math.ceil(x_train.shape[0] / Batch_Size)

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for di in range(100):
      for i in range(Batch_Count):
        xinput = x_train[i * Batch_Size : i * Batch_Size + Batch_Size]
        yinput = y_train[i * Batch_Size : i * Batch_Size + Batch_Size]
        session.run(operation, feed_dict={x_input: xinput, y_input: yinput})
      # session.run(operation, feed_dict={x_input: x_train, y_input: y_train})
    
      # if di % 100 is 0:
      print('The %d epoch, Accuracy is %f' % (di, session.run(accuracy, feed_dict={x_input: x_train, y_input: y_train})))
    print('The %d epoch, Accuracy is %f' % (di, session.run(accuracy, feed_dict={x_input: x_test, y_input: y_test})))    