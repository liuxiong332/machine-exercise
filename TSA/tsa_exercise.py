import numpy as np
import pandas as pd
import statsmodels
import statsmodels.tsa as smt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import itertools

def read_data():
  data = pd.read_csv('./series1.csv', index_col=0, parse_dates=[0])

  print(data)
  return data
  train, test = train_test_split(data, test_size=0.2)
  return train, test

def tsplot(data):
  fig = plt.figure(figsize=(14, 8))
  layout = (2, 2)
  ts_ax = plt.subplot2grid(layout, (0, 0))
  hist_ax = plt.subplot2grid(layout, (0, 1))
  acf_ax = plt.subplot2grid(layout, (1, 0))
  pacf_ax = plt.subplot2grid(layout, (1, 1))

  data.plot(ax=ts_ax)
  ts_ax.set_title('Data Series')

  data.plot(ax=hist_ax, kind='hist', bins=25)
  hist_ax.set_title('Histogram')

  statsmodels.graphics.tsaplots.plot_acf(data, lags=20, ax=acf_ax)
  statsmodels.graphics.tsaplots.plot_pacf(data, lags=20, ax=pacf_ax)

def find_args(data):
  p_range = range(0, 5)
  q_range = range(0, 5)
  d_range = range(1, 2)

  items = {}
  for p, q, d in  itertools.product(p_range, q_range, d_range):
    if p is 0 and q is 0 and d is 0:
      continue
    model = smt.arima_model.ARIMA(data, order=(p, d, q))
    try:
      results = model.fit()
    except ValueError as e:
      continue
    items['(%d,%d,%d)' % (p, d, q)] = results.bic

  res_series = pd.Series(items)
  argkey = np.argmin(res_series)
  print('The best params is %s, the bic is %f' % (argkey, res_series[argkey]))
  # The best params is (2,1,1), the bic is 373.407150

def train_params(params, data):
    model = smt.arima_model.ARIMA(data, order=params)
    results = model.fit()
    # 对残差进行 ACF分析，都在置信空间内，表示可以
    plt.figure()
    ax = plt.subplot(211)
    statsmodels.graphics.tsaplots.plot_acf(results.resid, ax=ax)

    # ax = plt.subplot(312)
    # statsmodels.graphics.tsaplots.plot_pacf(results.resid, ax=ax)

    ax = plt.subplot(212)
    statsmodels.graphics.gofplots.qqplot(results.resid, ax=ax)
  
def train_data():
  train = read_data()

  # tsplot(train)
  # find_args(train)
  train_params((2,1,1), train)