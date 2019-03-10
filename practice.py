import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsRegressor

home_data = pd.read_csv('Training.csv', header = 0)
home_data.columns = home_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
home_data = home_data.astype(float)

test_data = pd.read_csv('Testing2_.csv', header =0)
test_data.columns = test_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
test_data = test_data.astype(float)


answer_data = pd.read_csv('Answer - Testing2 - Student.csv', header =0)
answer_data.columns = answer_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
answer_data = answer_data.astype(float)

train_df = home_data
test_df = test_data


print(train_df.columns.values)

#make predictions with
x = ['space_ft2', 'bedroom' , 'bathroom', 'exterior', 'interior']

#predict
y = ['price']

knn = KNeighborsRegressor(algorithm='brute')

knn.fit(train_df[x], train_df[y])

predictions = knn.predict(test_df[x])

actual = answer_data[y]

mse = (((predictions - actual) ** 2).sum()) / len(predictions)

print(mse)





# def normalize(dataset):
#     dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))*20
#     dataNorm["zipcode"]=dataset["zipcode"]
#     return dataNorm
#
#
# def predict_price(new_listing_value, column1):
#     temp_df = train_df
#     temp_df['space_diff'] = np.abs(home_data[column1] - new_listing_value)
#     temp_df = temp_df.sort_values('space_diff')
#     knn = temp_df.price.iloc[:5]
#     predicted_price = knn.mean()
#     return (predicted_price)
#
# test_df['predicted_price'] = test_df.space_ft2.apply(predict_price, column1='space_ft2')
# # print(test_df)
#
# test_df['squared_error'] = ((test_df['predicted_price'] - answer_data.price )**(2))
# mse = test_df['squared_error'].mean()
# rmse = mse ** (1/2)
# # print(rmse)
#
# # for feature in ['space_ft2','bedroom','bathroom','housing_type']:
# #     test_df['predicted_price'] = test_df.space_ft2.apply(predict_price,column1=feature)
# #
# #     test_df['squared_error'] = (test_df['predicted_price'] - answer_data['price'])**(2)
# #     mse = test_df['squared_error'].mean()
# #     rmse = mse ** (1/2)
# #     print("RMSE for the {} column: {}".format(feature,rmse))
#
# def predict_price_multivariate(new_listing_value,feature_columns):
#     temp_df = train_df
#     temp_df['distance'] = distance.cdist(temp_df[feature_columns],[new_listing_value[feature_columns]])
#     temp_df = temp_df.sort_values('distance')
#     knn_5 = temp_df.price.iloc[:5]
#     predicted_price = knn_5.mean()
#     return(predicted_price)
#
# cols = ['space_ft2', 'bedroom', 'bathroom']
# norm_test_df['predicted_price'] = norm_test_df[cols].apply(predict_price_multivariate,feature_columns=cols,axis=1)
# norm_test_df['squared_error'] = (norm_test_df['predicted_price'] - answer_data['price'])**(2)
# mse = norm_test_df['squared_error'].mean()
# rmse = mse ** (1/2)
# print(rmse)
