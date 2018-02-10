import logistic
import numpy as np

def colic_test():
  with open('horseColicTraining.txt') as fr:
    data_mat, label_mat = [], []
    for line in fr.readlines():
      line_data = line.strip().split('\t')
      line_data = [float(d) for d in line_data]
      data_mat.append(line_data[:-1])
      label_mat.append(line_data[-1])
  weights = logistic.grad_ascend(data_mat, label_mat)
  print(weights)

  error_count = 0
  with open('horseColicTest.txt') as fr:
    lines = fr.readlines()
    for line in lines:
      line_data = line.strip().split('\t')
      line_data = [float(d) for d in line_data]
      label_val = np.dot(np.array(line_data[:-1]), weights)
      label = 1 if label_val >= 0 else 0
      if label != line_data[-1]:
        error_count += 1
  print('the error rate is ', error_count / len(lines))
  return error_count, len(lines)

def multi_test():
  errors, counts = 0, 0
  for i in range(10):
    error_count, all_count = colic_test()
    errors += error_count
    counts += all_count
  print('Average error rate is ', errors / counts)