import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot
import itertools
from imblearn.over_sampling import SMOTE

def calc_recall(x_train, y_train, c):
  kf = KFold(5)  # 5 folds verification
  recall_s = []
  for train_index, test_index in kf.split(x_train):
    lr = LogisticRegression(penalty='l2', C=c)
    lr.fit(x_train[train_index], y_train[train_index])
    y_predict = lr.predict(x_train[test_index])
    recall = recall_score(y_train[test_index], y_predict)
    recall_s.append(recall)
  recall_s = np.sum(recall_s) / len(recall_s)
  return recall_s

def find_best_c(x_train, y_train):
  c_values = [0.01, 0.1, 1, 10, 100]
  recall_values = []
  for c in c_values:
    recall_val = calc_recall(x_train, y_train, c)
    recall_values.append(recall_val)
    print('when c = ', c, ', the recall value is ', recall_val)
  max_index = np.argmax(recall_values)
  print('max best_c is ', c_values[max_index])
  return c_values[max_index]

def load_dataset():
  dataset = pd.read_csv('creditcard.csv')
  dataset = dataset.loc[:, dataset.columns.drop(['Time'])]
  scaler = StandardScaler()
  dataset['Amount'] = scaler.fit_transform(dataset['Amount'].values.reshape(-1, 1))

  # undersample_train(dataset)
  # undersample_train_proba(dataset)
  oversample_train_proba(dataset)

def plot_confusion_matrix(y_true, y_predict):
  cm = confusion_matrix(y_true, y_predict)
  pyplot.imshow(cm, cmap='Blues')
  pyplot.xticks(np.arange(2), [0, 1])
  pyplot.yticks(np.arange(2), [0, 1])
  pyplot.xlabel('True label')
  pyplot.ylabel('Predict label')
  pyplot.colorbar()
  for i, j in itertools.product(np.arange(2), np.arange(2)):
    pyplot.text(i, j, cm[j, i])

# 下采样训练
def undersample_train(dataset):
  dataset = undersample_select(dataset)
  labels = dataset.loc[:, 'Class']
  trdata = dataset[dataset.columns.drop(['Class'])]
  x_train, x_test, y_train, y_test = train_test_split(trdata, labels, test_size=0.3)
  
  best_c = find_best_c(x_train.values, y_train.values)
  lr = LogisticRegression(penalty='l2', C=best_c)
  lr.fit(x_train, y_train)
  y_predict = lr.predict(x_test)
  recall_val = recall_score(y_test, y_predict)
  print('recall_val:', recall_val)
  plot_confusion_matrix(y_test, y_predict)

# 下采样训练，且让threshold值可变
def undersample_train_proba(dataset):
  dataset = undersample_select(dataset)
  labels = dataset.loc[:, 'Class']
  trdata = dataset[dataset.columns.drop(['Class'])]
  x_train, x_test, y_train, y_test = train_test_split(trdata, labels, test_size=0.3)
  
  best_c = find_best_c(x_train.values, y_train.values)
  lr = LogisticRegression(penalty='l2', C=best_c)
  lr.fit(x_train, y_train)
  y_predict_prob = lr.predict_proba(x_test)

  threshold_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  for i, threshold in enumerate(threshold_vals):
    y_predict = y_predict_prob[:, 1] > threshold
    pyplot.subplot(3, 3, i + 1)
    plot_confusion_matrix(y_test, y_predict)
    recall_val = recall_score(y_test, y_predict)
    precision_val = precision_score(y_test, y_predict)
    print('When threshold value is %f, the recall value is %f, the precision value is %f' % 
      (threshold, recall_val, precision_val))

# 使用SMOTE过采样
def oversample_train_proba(dataset):
  labels = dataset.loc[:, 'Class']
  trdata = dataset[dataset.columns.drop(['Class'])]
  x_train, x_test, y_train, y_test = train_test_split(trdata, labels, test_size=0.3)
  
  sm = SMOTE()
  x_train, y_train = sm.fit_sample(x_train, y_train)
  best_c = find_best_c(x_train, y_train)
  lr = LogisticRegression(penalty='l2', C=best_c)
  lr.fit(x_train, y_train)
  y_predict_prob = lr.predict_proba(x_test)

  threshold_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  for i, threshold in enumerate(threshold_vals):
    y_predict = y_predict_prob[:, 1] > threshold
    pyplot.subplot(3, 3, i + 1)
    plot_confusion_matrix(y_test, y_predict)
    recall_val = recall_score(y_test, y_predict)
    precision_val = precision_score(y_test, y_predict)
    print('When threshold value is %f, the recall value is %f, the precision value is %f' % 
      (threshold, recall_val, precision_val))

def undersample_select(dataset):
  pcount = len(dataset[dataset['Class'] == 1])
  nlabels = dataset[dataset['Class'] == 0]
  n_indexs = np.random.choice(nlabels.index, pcount)
  
  undersample_ds = np.concatenate([dataset[dataset['Class'] == 1], dataset.loc[n_indexs]])
  return pd.DataFrame(undersample_ds, columns=dataset.columns)