import numpy as np 
import pandas as pd 
import math 
import scipy.spatial as spatial
import sklearn.preprocessing as preprocessing
import sklearn.neighbors as neighbors
import sklearn.metrics as metrics

def knn_train_by_columns(test_data, train_set, columns, label_col, k=5):
    filter_test = test_data[columns]
    def calc_square(series):
        square_list = np.abs(filter_test - series[columns])
        return pd.Series({ 'distance': square_list.sum(), 'price': series[label_col] })

    new_set = pd.DataFrame({ 'distance': spatial.distance.cdist(train_set[columns], [test_data[columns]]).flatten(), 'price': train_set.price })
    return pd.Series({ 'predict_price': new_set.sort_values('distance').price.iloc[:5].mean(), 'price': test_data.price })
    # dis_result = train_set.apply(calc_square, axis=1)
    train_set['distance'] = np.abs(train_set[columns] - test_data[columns])[columns]
    predict_price = train_set.sort_values('distance').price.iloc[:5].mean()
    return predict_price

    '''
    predict_val = dis_result.sort_values('distance')['price'][0:k].mean()
    print('price:', predict_val, test_data[label_col])
    return pd.Series({ 'predict': predict_val, 'price': test_data[label_col] })
    '''

def calc_rmse(test_result):
    square_list = (test_result['predict_price'] - test_result['price']) ** 2
    print('The RMSE is ', square_list.mean() ** (1 / 2))

def knn_train():
    features = ['accommodates', 'beds']
    all_features = features + ['price']
    label_col = 'price'
    dataset = pd.read_csv('kNN/listings.csv')
    dataset = dataset[all_features]
    dataset = dataset.sample(frac=1)  # Make the data row random
    # dataset['price'] = dataset['price'].apply(lambda x: float(x[1:].replace(',', '')))
    dataset['price'] = dataset['price'].str.replace(r',|\$', '').astype(float)
    dataset = dataset.dropna()

    dataset[all_features] = preprocessing.StandardScaler().fit_transform(dataset[all_features])
    train_count = 2792 # math.floor(len(dataset) * 0.7)
    train_set = dataset.iloc[:train_count]
    test_set = dataset.iloc[train_count:]
    result_set = test_set.apply(knn_train_by_columns, axis=1, train_set=train_set, columns=features, label_col=label_col)
    calc_rmse(result_set)

def sklearn_train():
    features = ['accommodates', 'beds']    
    all_features = features + ['price']
    dataset = pd.read_csv('kNN/listings.csv')

    dataset = dataset[all_features]
    dataset['price'] = dataset['price'].str.replace(r',|\$', '').astype(float)
    dataset = dataset.dropna()
    dataset[all_features] = preprocessing.StandardScaler().fit_transform(dataset[all_features])

    train_count = 2792 # math.floor(len(dataset) * 0.7)
    train_set = dataset.iloc[:train_count]
    test_set = dataset.iloc[train_count:]

    knn = neighbors.KNeighborsRegressor()
    knn.fit(train_set[features], train_set.price)
    predict_vals = knn.predict(test_set[features])

    print('The rmse is ', metrics.mean_squared_error(predict_vals, test_set.price) ** 0.5)

if __name__ == '__main__':
    import time
    start_time = time.time()
    knn_train()
    # sklearn_train()
    end_time = time.time()
    print('spent time %f s' % (end_time - start_time))
