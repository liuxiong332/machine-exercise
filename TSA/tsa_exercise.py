import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

def read_data():
  data = pd.read_csv('./series1.csv', index_col=0, parse_dates=[0])

  train, test = train_test_split(data, test_size=0.2)
  print(train)
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

def train_data():
  train, test = read_data()

  tsplot(train)