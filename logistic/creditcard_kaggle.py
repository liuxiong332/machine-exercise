import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot
import itertools
from imblearn.over_sampling import SMOTE

def calc_auc(x_train, y_train, c):
  kf = StratifiedKFold(5)  # 5 folds verification
  scores = []
  for train_index, test_index in kf.split(x_train):
    print(y_train[test_index])
    lr = LogisticRegression(penalty='l2', C=c)
    lr.fit(x_train[train_index], y_train[train_index])
    y_predict = lr.predict(x_train[test_index])
    auc_score = roc_auc_score(y_train[test_index], y_predict)
    scores.append(auc_score)
  scores = np.sum(scores) / len(scores)
  return scores

def find_best_c(x_train, y_train):
  c_values = [0.01, 0.1, 1, 10, 100]
  recall_values = []
  for c in c_values:
    recall_val = calc_auc(x_train, y_train, c)
    recall_values.append(recall_val)
    print('when c = ', c, ', the auc value is ', recall_val)
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
  # plot_confusion_matrix(y_test, y_predict)

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
  ks_values = []
  for i, threshold in enumerate(threshold_vals):
    y_predict = y_predict_prob[:, 1] > threshold
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    ks_val = tpr - fpr
    ks_values.append(ks_val)
    print('When tpr is %f, the fpr is %f, ks val is %f' % (tpr, fpr, ks_val))
  threshold = threshold_vals[np.argmax(ks_values)]
  y_predict = y_predict_prob[:, 1] > threshold
  print('recall val is:', recall_score(y_test, y_predict))

# 使用SMOTE过采样
def oversample_train_proba(dataset):
  labels = dataset.loc[:, 'Class']
  trdata = dataset[dataset.columns.drop(['Class'])]
  x_train, x_test, y_train, y_test = train_test_split(trdata, labels, test_size=0.3)
  
  sm = SMOTE()
  x_train, y_train = sm.fit_sample(x_train, y_train)
  print('train label 0 count %d, train label 1 count %d' % (len(y_train == 0)), len(y_train == 1))

  best_c = find_best_c(x_train, y_train)
  lr = LogisticRegression(penalty='l2', C=best_c)
  lr.fit(x_train, y_train)
  y_predict_prob = lr.predict_proba(x_test)

  threshold_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  ks_values = []
  for i, threshold in enumerate(threshold_vals):
    y_predict = y_predict_prob[:, 1] > threshold
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    ks_val = tpr - fpr
    ks_values.append(ks_val)
    print('When tpr is %f, the fpr is %f, ks val is %f' % (tpr, fpr, ks_val))
  threshold = threshold_vals[np.argmax(ks_values)]
  y_predict = y_predict_prob[:, 1] > threshold
  print('recall val is:', recall_score(y_test, y_predict))
  
def undersample_select(dataset):
  pcount = len(dataset[dataset['Class'] == 1])
  nlabels = dataset[dataset['Class'] == 0]
  n_indexs = np.random.choice(nlabels.index, pcount)
  
  undersample_ds = np.concatenate([dataset[dataset['Class'] == 1], dataset.loc[n_indexs]])
  return pd.DataFrame(undersample_ds, columns=dataset.columns)