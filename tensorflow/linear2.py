import numpy as np
import tensorflow as tf

def gen_data():
  xdata = []
  ydata = []
  for _ in range(1000):
    xval = np.random.randn()
    xdata.append(xval)
    yval = xval * 0.4 + 1.2 + np.random.randn()
    ydata.append(yval)
  return xdata, ydata

def train_data(xdata, ydata):
  w = tf.Variable(np.random.randn())
  b = tf.Variable(np.random.randn())
  y = w * xdata + b
  loss = tf.reduce_mean(tf.square(y - ydata))

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  opt = optimizer.minimize(loss)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
      sess.run(opt)
      if i % 100 == 0:
        print('w is %f, b is %f, loss is %f' % (w.eval(), b.eval(), loss.eval()))

def run_train():
  train_data(*gen_data())