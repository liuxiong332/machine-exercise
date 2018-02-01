import numpy as np 
import pandas as pd 
import math 

def knn_train_by_columns(test_data, train_set, columns, label_col, k=5):
    filter_test = test_data[columns]
    def calc_square(series):
        square_list = np.abs(filter_test - series[columns])
        return pd.Series({ 'distance': square_list.sum(), 'price': series[label_col] })

    # dis_result = train_set.apply(calc_square, axis=1)
    train_set['distance'] = np.abs(train_set[columns] - test_data[columns])
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
    features = ['accommodates']
    label_col = 'price'
    dataset = pd.read_csv('kNN/listings.csv')
    dataset.dropna()
    dataset = dataset.sample(frac=1)  # Make the data row random
    dataset['price'] = dataset['price'].apply(lambda x: float(x[1:].replace(',', '')))
    train_count = 2792 # math.floor(len(dataset) * 0.7)
    train_set = dataset[:train_count]
    test_set = dataset[train_count:]
    test_set['predict_price'] = test_set.apply(knn_train_by_columns, axis=1, train_set=train_set, columns=features, label_col=label_col)
    calc_rmse(test_set)

if __name__ == '__main__':
    knn_train()