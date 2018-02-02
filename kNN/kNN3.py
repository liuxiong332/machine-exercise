import pandas as pd
import numpy as np
import time

start_time = time.time()

features = ['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']

dc_listings = pd.read_csv('kNN/listings.csv')
dc_listings['price'] = dc_listings.price.str.replace("\$|,",'').astype(float)

train_df = dc_listings.copy().iloc[:2792]
test_df = dc_listings.copy().iloc[2792:]

def predict_price(new_listing_value,feature_column):
    temp_df = train_df
    temp_df['distance'] = np.abs(dc_listings[feature_column] - new_listing_value)
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return(predicted_price)

test_df['predicted_price'] = test_df.accommodates.apply(predict_price,feature_column='accommodates')

test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
mse = test_df['squared_error'].mean()
rmse = mse ** (1/2)
print(rmse)

end_time = time.time()
print('spent time %f s' % (end_time - start_time))