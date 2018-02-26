import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot
from sklearn.preprocessing import PolynomialFeatures

def load_dataset(filename):
  data = np.loadtxt(filename, delimiter=',')
  X = np.c_[np.ones(len(data)), data[:, 0:2]]
  Y = data[:, 2]
  return X, Y

def polynomial_dataset(X):
  pf = PolynomialFeatures(6)
  return pf.fit_transform(X[:, 1:])

def sigmoid(z):
  return 1 / (1 + np.exp(-z)) 

def cost(w, X, Y):
  m = len(X)
  sigval = sigmoid(np.dot(X, w))
  return -1 / m * (np.dot(Y, np.log(sigval)) + np.dot(1 - Y, np.log(1 - sigval)))

def regularization_cost(w, reg, X, Y):
  m = len(X)
  sigval = sigmoid(np.dot(X, w))
  regularization_val = reg / (2 * m) * np.sum(w[1:] ** 2)
  j = -1 / m * (np.dot(Y, np.log(sigval)) + np.dot(1 - Y, np.log(1 - sigval))) + regularization_val
  if np.isnan(j):
    return np.inf 
  return j

def gradient(w, X, Y):
  m = len(X)
  return -1 / m * np.dot(X.T, Y - sigmoid(np.dot(X, w)))

def regularization_gradient(w, reg, X, Y):
  m = len(X)
  return -1 / m * np.dot(X.T, Y - sigmoid(np.dot(X, w))) + reg / m * np.r_[[0], w[1:]]

def draw_dots(ax, X, Y, res):
  negative_points = X[Y == 0][:, 1:]
  positive_points = X[Y == 1][:, 1:]
  ax.scatter(negative_points[:, 0], negative_points[:, 1], s=30, c='red', marker='x')
  ax.scatter(positive_points[:, 0], positive_points[:, 1], s=30, c='blue', marker='o')

def draw_decision_boundary(ax, X, Y, res):
  x_min, x_max = np.min(X[:, 1]), np.max(X[:, 1])
  y_min, y_max = np.min(X[:, 2]), np.max(X[:, 2])
  xx, xy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
  height = np.dot(np.c_[np.ones(len(xx) ** 2), xx.ravel(), xy.ravel()], res.x).reshape(len(xx), -1)
  ax.contour(xx, xy, height, [0.5], colors='green')

def draw_reg_decision_boundary(ax, X, Y, res):
  x_min, x_max = np.min(X[:, 1]), np.max(X[:, 1])
  y_min, y_max = np.min(X[:, 2]), np.max(X[:, 2])
  xx, xy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
  pf = PolynomialFeatures(6)
  height = np.dot(pf.fit_transform(np.c_[xx.ravel(), xy.ravel()]), res.x).reshape(xx.shape)
  ax.contour(xx, xy, height, [0.2, 0.5], colors='green')

# 计算data1的 thlta，并且画出data1的决策边界
def show_data1_logistic():
  X, Y = load_dataset('mldata1.txt')
  init_thelta = np.zeros(X.shape[1])
  res = minimize(cost, init_thelta, (X, Y), jac=gradient, options={'maxiter': 5000})
  print(res)
  fig, ax = pyplot.subplots()
  draw_dots(ax, X, Y, res)
  draw_decision_boundary(ax, X, Y, res)

def show_data2_logistic():
  X, Y = load_dataset('mldata2.txt')
  X = polynomial_dataset(X)
  init_thelta = np.zeros(X.shape[1])
  fig, axes = pyplot.subplots(1, 3)
  for i, reg in enumerate([0, 0.5, 10]):
    res = minimize(regularization_cost, init_thelta, (reg, X, Y), jac=regularization_gradient, options={'maxiter': 3000})
    print(res)
    draw_dots(axes[i], X, Y, res)
    draw_reg_decision_boundary(axes[i], X, Y, res)