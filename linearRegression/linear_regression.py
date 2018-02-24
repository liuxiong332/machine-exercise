import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.linear_model import LinearRegression

def load_dataset():
    data = np.loadtxt('linear_regression_data1.txt', delimiter=',')
    X = np.c_[np.ones(data.shape[0]), data[:, 0]]
    Y = data[:, 1]
    return X, Y

def compute_cost(X, W, Y):
    return 1 / (2 * len(Y)) * np.sum((np.dot(X, W) - Y) ** 2)

def gradient_descent(X, Y):
    w = np.zeros(X.shape[1])    # 初始[0, 0]
    m = len(Y)
    Delta_Count = 1000
    alpha = 0.01
    costs = []
    for i in range(Delta_Count):
        w = w - alpha * 1 / m * np.dot(X.T, np.dot(X, w) - Y)
        costs.append(compute_cost(X, w, Y))
    return w, costs

def draw_linear():
    X, Y = load_dataset()
    w, costs = gradient_descent(X, Y)
    pyplot.subplot(111)
    pyplot.scatter(X[:, 1], Y, s=30, c='red', marker='x')

    xrange = np.arange(0, 30)
    yrange = xrange * w[1] + w[0]
    pyplot.plot(xrange, yrange, c='blue')

    lr = LinearRegression()
    lr.fit(X[:, 1].reshape(-1, 1), Y)
    ylrange = xrange * lr.coef_ + lr.intercept_
    pyplot.plot(xrange, ylrange, c='green')