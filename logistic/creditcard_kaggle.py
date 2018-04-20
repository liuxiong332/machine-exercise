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
  kf = StratifiedKFold(5, shuffle=True)  # 5 folds verification
  scores = []
  for train_index, test_index in kf.split(x_train, y_train):
    lr = LogisticRegression(penalty='l2', C=c)
    lr.fit(x_train[train_index], y_train[train_index])
    y_predict = lr.predict_proba(x_train[test_index])

    y_true = y_train[test_index]
    # print('train label 0 count %d, train label 1 count %d' % (len(y_true == 0), len(y_true == 1)))
    auc_score = roc_auc_score(y_train[test_index], y_predict[:, 1])
    # print('when c is %f, the auc score is %f' %(c, auc_score))
    scores.append(auc_score)
  scores = np.sum(scores) / len(scores)
  print('When C=%f, the last AUC score=%f' %(c, scores))
  return scores

def find_best_c(x_train, y_train):
  c_values = [0.01, 0.1, 1, 10, 100]
  recall_values = []
  for c in c_values:
    recall_val = calc_auc(x_train, y_train, c)
    recall_values.append(recall_val)
    # print('when c = ', c, ', the auc value is ', recall_val)
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

def calc_kfold_ks(x_train, y_train, c, threshold):
  kf = StratifiedKFold(3, shuffle=True)
  scores = []
  for train_index, test_index in kf.split(x_train, y_train):
    lr = LogisticRegression(penalty='l2', C=c)
    lr.fit(x_train[train_index], y_train[train_index])
    predict_proba = lr.predict_proba(x_train[test_index])
    score = calc_ks(y_train[test_index], predict_proba, threshold)
    # print('ks val: ', score)
    scores.append(score)
  ks_val = np.mean(scores)
  print('When thresold is %f, the ks value is %f' %(threshold, ks_val))
  return ks_val

def calc_ks(y_test, y_predict_prob, threshold):
  y_predict = y_predict_prob[:, 1] > threshold
  (tn, fp), (fn, tp) = confusion_matrix(y_test, y_predict, labels=[0, 1])
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  ks_val = tpr - fpr
  return ks_val

def find_threshold(x_train, y_train, c):
  threshold_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  ks_values = []
  for i, threshold in enumerate(threshold_vals):
    ks_val = calc_kfold_ks(x_train, y_train, c, threshold)
    ks_values.append(ks_val)
  threshold = threshold_vals[np.argmax(ks_values)]
  print('The last threshold is ', threshold)
  return threshold

# 使用SMOTE过采样
def oversample_train_proba(dataset):
  labels = dataset.loc[:, 'Class']
  trdata = dataset[dataset.columns.drop(['Class'])]
  x_train, x_test, y_train, y_test = train_test_split(trdata, labels, test_size=0.3)
  
  sm = SMOTE()
  x_train, y_train = sm.fit_sample(x_train, y_train)
  # print('train label 0 count %d, train label 1 count %d' % (len(y_train == 0), len(y_train == 1)))

  best_c = find_best_c(x_train, y_train)
  print('the best c is %f' % best_c)

  threshold = find_threshold(x_train, y_train, best_c)

  lr = LogisticRegression(penalty='l2', C=best_c)
  lr.fit(x_train, y_train)
  y_predict_prob = lr.predict_proba(x_test)
  y_predict = y_predict_prob[:, 1] > threshold
  print('recall val is:', recall_score(y_test, y_predict))
  