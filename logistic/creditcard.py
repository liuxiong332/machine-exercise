import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

def load_dataset():
  dataset = pd.read_csv('creditcard.csv')
  labels = dataset.loc[:, 'Class']
  feature_data = dataset.loc[:, dataset.columns.drop(['Class', 'Time'])]
  scaler = StandardScaler()
  feature_data['Amount'] = scaler.fit_transform(feature_data['Amount'].values.reshape(-1, 1))
  x_train, x_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.3)
  
  kf = KFold(5)
  recall_s = []
  for train_index, test_index in kf.split(x_train):
    lr = LogisticRegression(penalty='l2', C=1)
    lr.fit(x_train[train_index], y_train[train_index])
    y_predict = lr.predict(x_train[test_index])
    recall = recall_score(y_train[test_index], y_predict)
    recall_s.append(recall)
  recall_s = np.sum(recall_s) / len(recall_s)

def train_undersample():
  pass