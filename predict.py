import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

test_data = pd.read_csv('./data/car_prices_test.csv')
test_data.dropna(inplace=True)

y_test = test_data['sellingprice'].astype(np.float32)
X_test = test_data[['year', 'condition', 'transmission']]

scaler_y = MinMaxScaler()
scaler_y.fit(y_test.values.reshape(-1, 1))

scaler_X = MinMaxScaler()
X_test['condition'] = scaler_X.fit_transform(X_test[['condition']])
X_test = pd.get_dummies(X_test, columns=['transmission'])

model = tf.keras.models.load_model('./car_prices_predict_model.h5')

y_pred_scaled = model.predict(X_test)

y_pred = scaler_y.inverse_transform(y_pred_scaled)

y_pred_df = pd.DataFrame(y_pred, columns=['PredictedSellingPrice'])
y_pred_df.to_csv('predicted_selling_prices.csv', index=False)
