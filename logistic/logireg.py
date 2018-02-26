import numpy as np
import pandas as pd
import random
import functools
from sklearn import preprocessing as pp

STOP_ITERATE_SIZE = 1
STOP_COST = 2
STOP_GRAD = 3

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def get_data():
  data = pd.read_csv('LogiReg_data.txt', header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
  data.insert(2, 'ones', 1)
  # np.random.shuffle(data)  # 就地打乱顺序  
  all_data = data[['Exam 1', 'Exam 2', 'ones']].as_matrix()
  all_label = data['Admitted'].as_matrix()

  all_data[:, 0:2] = pp.scale(all_data[:, 0:2])
  # all_label = pp.scale(all_label)
  
  count = int(len(all_data) * 0.8)
  train_data = all_data[:count]
  train_label = all_label[:count]

  test_data = all_data[count:]
  test_label = all_label[count:]
  return train_data, train_label, test_data, test_label

def stop_by_iterate_count(max_count, **kwargs):
  return kwargs['train_count'] < max_count  # 小于1万次迭代

def stop_by_costs(delta_cost, **kwargs):
  costs = kwargs['costs']
  if len(costs) > 2:
    isOK = np.abs(costs[-1] - costs[-2]) < delta_cost
    return not isOK
  return True

def stop_by_grad(grad_val, **kwargs): # 当梯度值小于某一个值时，停止迭代
  return np.linalg.norm(kwargs['grad']) > grad_val

def calc_cost(w, train_data, label):
  hx = sigmoid(np.dot(train_data, w))
  return np.sum(-label * np.log(hx) - (1 - label) * np.log(1 - hx)) / len(train_data)

def train(train_data, train_label, strategy_checker, batch_size, alpha = 0.001):
  cols = train_data.shape[1]
  row_count = len(train_data)
  w = np.zeros(cols)
  train_count = 0  # 训练迭代次数
  costs = [calc_cost(w, train_data, train_label)]  # 每次迭代的损失值数组
  start_index = 0
  while True:
    # row_indexs = random.sample(range(row_count), batch_size)
    calc_trains = train_data[start_index : start_index + batch_size]
    calc_labels = train_label[start_index : start_index + batch_size]
    start_index += batch_size
    start_index = start_index % row_count
    # errors = real_y - predict_y(sigmoid(w * x))
    errors = calc_labels - sigmoid(np.dot(calc_trains, w))
    grad = alpha * np.dot(calc_trains.T, errors) / batch_size  # 本次梯度上升值
    w = w + grad
    train_count += 1
    costs.append(calc_cost(w, calc_trains, calc_labels))

    if not strategy_checker(train_count=train_count, costs=costs, grad=grad):
      break
  return w, costs

def calc_error_rate(w, test_data, test_label):
  predict_y = sigmoid(np.dot(test_data, w)) > 0.5 # 大于0.5 是1， 小于则是0
  error_rate = np.sum(np.abs(test_label - predict_y)) / len(test_data)
  print('The error rate is %f, w value is %s' % (error_rate, w)) 
    
def run_itercount_test(ax1):
  train_data, train_label, test_data, test_label = get_data()
  w, costs = train(train_data, train_label, functools.partial(stop_by_iterate_count, 6000), len(train_data))
  calc_error_rate(w, test_data, test_label)
  # w_results = np.linalg.norm(w_results, axis=1)
  ax1.plot(np.arange(len(costs)), costs)
  ax1.set_xlabel('train count')
  ax1.set_ylabel('costs')

def run_cost_test(ax1):
  train_data, train_label, test_data, test_label = get_data()
  w, costs = train(train_data, train_label, functools.partial(stop_by_costs, 0.00000001), len(train_data))
  calc_error_rate(w, test_data, test_label)
  # w_results = np.linalg.norm(w_results, axis=1)
  ax1.plot(np.arange(len(costs)), costs)
  ax1.set_xlabel('train count')
  ax1.set_ylabel('costs')

def run_test():
  import matplotlib.pyplot as pyplot
  fig, (ax1) = pyplot.subplots(1, 1)

  train_data, train_label, test_data, test_label = get_data()
  w, costs = train(train_data, train_label, functools.partial(stop_by_grad, 0.00002), len(train_data))
  calc_error_rate(w, test_data, test_label)
  # w_results = np.linalg.norm(w_results, axis=1)
  ax1.plot(np.arange(len(costs)), costs)
  ax1.set_xlabel('train count')
  ax1.set_ylabel('costs')

if __name__ == '__main__':
  run_test()