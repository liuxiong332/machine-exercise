import numpy as np
import numpy.matlib as matlib
import math

def load_dataset():
  data_mat = []
  label_mat = []
  with open('testSet.txt') as fp:
    for line in fp.readlines():
      line_vals = line.strip().split()
      data_mat.append([float(line_vals[0]), float(line_vals[1]), 1])
      label_mat.append(int(line_vals[2]))
  return data_mat, label_mat

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def grad_ascend2(data_in, label_in):
  data_mat = matlib.mat(data_in)
  label_mat = matlib.mat(label_in).transpose()

  m, n = matlib.shape(data_mat)
  weights = matlib.ones(n).T
  alpha = 0.001
  max_cycle = 500
  for _ in range(max_cycle):
    calc_labels = sigmoid(data_mat * weights) # sigmoid(XW)
    error = label_mat - calc_labels
    weights = weights + alpha * data_mat.T * error
  return weights

def grad_ascend(data_in, label_in):
  data_mat = np.array(data_in)
  label_mat = np.array(label_in)
  m, n = matlib.shape(data_mat)
  weights = np.ones(n)
  alpha = 0.001
  max_cycle = 150
  for i in range(max_cycle):
    data_index = list(range(m))    
    for j in range(m):
      random_index = int(np.random.uniform(0, len(data_index)))
      alpha = 4 / (1 + i + j) + 0.01

      # print(data_mat[random_index] * weights)
      x = np.sum(data_mat[random_index] * weights)
      if x == 0: print(x)
      calc_labels = sigmoid(np.sum(data_mat[random_index] * weights)) # sigmoid(XW)
      error = label_mat[random_index] - calc_labels
      weights = weights + alpha * data_mat[random_index] * error
      # del data_index[random_index]
  return weights

def plot_bestfit():
  import matplotlib.pyplot as pyplot
  data_in, label_in = load_dataset()
  weights = grad_ascend(data_in, label_in)
  fig = pyplot.figure()
  ax = fig.add_subplot(111)
  data_arr = np.array(data_in)
  dots1x, dots1y, dots0x, dots0y = [], [], [], []
  for i in range(data_arr.shape[0]):
    if label_in[i] == 1:
      dots1x.append(data_arr[i, 0])
      dots1y.append(data_arr[i, 1])
    else:
      dots0x.append(data_arr[i, 0])
      dots0y.append(data_arr[i, 1])
  partx = np.array([-3, 3])
  party = (-weights[2] - weights[0] * partx) / weights[1]   
  print(partx, party)
  ax.scatter(dots0x, dots0y, s=30, c='red')
  ax.scatter(dots1x, dots1y, s=30, c='green')
  ax.plot(partx, party)
  ax.set_xlabel('X1')
  ax.set_ylabel('X2')
  fig.show()
  