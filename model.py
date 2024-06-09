import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

train_data = pd.read_csv('./data/car_prices_train.csv')

train_data.dropna(inplace=True)

y_train = train_data['sellingprice'].astype(np.float32)

X_train = train_data[['year', 'condition', 'transmission']]

scaler_x = MinMaxScaler()
X_train['condition'] = scaler_x.fit_transform(X_train[['condition']])

scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_train = pd.get_dummies(X_train, columns=['transmission'])

model = Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=20, batch_size=32)

model.save('./car_prices_predict_model.h5')
