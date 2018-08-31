import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.tsa as smt
import matplotlib.pyplot as plt 

def read_data():
  data = pd.read_csv('./T10yr.csv', index_col=0, parse_dates=[0])
  data = data['Close'].resample('W-MON').mean()
  # print(data['Close'])
  train_len = int(len(data) * 0.7)
  train_data, test_data = data[0:train_len], data[train_len:-1]
  return train_data, test_data

def survey_data(data):
  plt.figure()
  ax1 = plt.subplot(211)
  data.plot(ax=ax1)
  ax2 = plt.subplot(212)
  data.diff().plot(ax=ax2)

def survey_acf(data):
  plt.figure()
  ax1 = plt.subplot(211)
  sm.graphics.tsaplots.plot_acf(data, lags=20, ax=ax1)
  ax2 = plt.subplot(212)
  sm.graphics.tsaplots.plot_pacf(data, lags=20, ax=ax2)

def train_model(data, test_data, params):
  results = smt.arima_model.ARIMA(data, order=params, freq='W-MON').fit()
  test_index = test_data.index
  pred = results.predict(test_index[0], test_index[-1], dynamic=True, typ='levels')
  print(pred)
  return pred

def plot_predict(test_data, pred):
  plt.figure()
  test_data.plot()
  pred.plot()

def main():
  train_data, test_data = read_data()
  print(test_data)
  # survey_data(train_data)
  # train_diff = train_data.diff().dropna()
  # survey_acf(train_diff)
  pred = train_model(train_data, test_data, (1,1,1))
  plot_predict(test_data, pred)
