import numpy as np
import pandas as pd
import pickle 
import xgboost as xgb

dtrain = xgb.DMatrix('./input/agaricus.txt.train')
dtest = xgb.DMatrix('./input/agaricus.txt.test')

param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
watchlist = [(dtest, 'test'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, 2, watchlist)

predicts = bst.predict(dtest)

label = dtest.get_label()
pred_result = predicts > 0.5

error_rate = np.sum([pred_result[i] != label[i] for i in range(len(label))]) / len(label)
print('Error rate is %f' % error_rate)