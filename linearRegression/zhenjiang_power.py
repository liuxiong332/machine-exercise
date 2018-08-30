import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

def load_dateset():
  data1 = pd.read_csv('./input/zhenjiang_power.csv')
  data2 = pd.read_csv('./input/zhenjiang_power_9.csv')
  return pd.concat([data1, data2])

def gen_testdata():
  date_index = pd.date_range('2016-10-1', periods=31, freq='D')
  test_df = pd.DataFrame({ 'record_date': date_index })
  test_df['power_consumption'] = 0
  return test_df

def merge_data(data):
  data = pd.concat([data, gen_testdata()]) 

  data = data[['record_date', 'power_consumption']]
  enddate = data.groupby('record_date').sum().reset_index()
  enddate['record_date'] = pd.to_datetime(enddate['record_date'])
  return enddate

def week_of_month(date):
  return date.day // 7

def period_of_month(date):
  if date.day in range(1, 11):
    return 1
  elif date.day in range(11, 21):
    return 2
  else:
    return 3

def extract_features(data):
  data['year'] = data['record_date'].apply(lambda d: d.year)
  data['month'] = data['record_date'].apply(lambda d: d.month)
  data['day'] = data['record_date'].apply(lambda d: d.day)
  data['week'] = data['record_date'].apply(lambda d: d.week)
  data['wofm'] = data['record_date'].apply(week_of_month)
  data['pofm'] = data['record_date'].apply(period_of_month)
  data['hofm'] = data['record_date'].apply(lambda d: 0 if d.day <= 15 else 1)
  data['dow'] = data['record_date'].apply(lambda d: d.dayofweek)
  data['weekend'] = data['record_date'].apply(lambda d: d.dayofweek >= 5)
  data['sat'] = data['record_date'].apply(lambda d: d.dayofweek == 5)
  data['sun'] = data['record_date'].apply(lambda d: d.dayofweek == 6)
  data['festival'] = data['record_date'].apply(lambda d: d.month == 10 and d.day < 8)

def train():
  data = load_dateset()
  data = merge_data(data)
  extract_features(data)
  columns = ['year', 'month', 'day', 'week', 'wofm', 'pofm', 'hofm', 'dow', 'weekend', 'sat', 'sun', 'festival']
  dummy_data = pd.get_dummies(data, columns=columns)

  test_data = dummy_data[dummy_data['record_date'] >= '2016-10-1']
  train_data = dummy_data[dummy_data['record_date'] < '2016-10-1']

  ridge_cv = RidgeCV(alphas=[0.2,0.5,0.8], cv=5)
  train_X = train_data.drop(['record_date', 'power_consumption'], axis=1)
  train_Y = train_data['power_consumption']
  ridge_cv.fit(train_X, train_Y)
  print(ridge_cv.score(train_X, train_Y))

  test_data['power_consumption'] = ridge_cv.predict(test_data.drop(['record_date', 'power_consumption'], axis=1))

train()