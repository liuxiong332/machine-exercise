import numpy as np
import numpy.matlib as matlib

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

def grad_ascend(data_in, label_in):
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

