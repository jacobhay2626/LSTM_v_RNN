import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout, Flatten

import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))

# We will use 50 data to predict the 51st data entry.

bit_data = pd.read_csv("/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")
bit_data["date"] = pd.to_datetime(bit_data["Timestamp"], unit="s").dt.date
group = bit_data.groupby("date")
data = group['Close'].mean()

# Describing the data

data.shape

data.isnull().sum()

data.head()

# Looking to predict daily close price

close_train = data.iloc[:len(data) - 50]
close_test = data.iloc[len(close_train):]

# set values between 0-1 in order to avoid domination of high values

close_train = np.array(close_train)
close_train = close_train.reshape(close_train.shape[0], 1)
scaler = MinMaxScaler(feature_range=(0, 1))
close_scaled = scaler.fit_transform(close_train)


# Each of the 50 data points as x_train, and the 51st as y_train

timestep = 50
x_train = []
y_train = []

for i in range(timestep, close_scaled.shape[0]):
    x_train.append(close_scaled[i - timestep:i, 0])
    y_train.append(close_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
print("x_train shape= ", x_train.shape)
print("y_train shape= ", y_train.shape)

# RNN

regressor = Sequential()

regressor.add(SimpleRNN(128, activation='relu', return_sequences=True, input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.25))

regressor.add(SimpleRNN(256, activation='relu', return_sequences=True))
regressor.add(Dropout(0.25))

regressor.add(SimpleRNN(512, activation='relu', return_sequences=True))
regressor.add(Dropout(0.25))

regressor.add(SimpleRNN(256, activation='relu', return_sequences=True))
regressor.add(Dropout(0.25))

regressor.add(SimpleRNN(128, activation='relu', return_sequences=True))
regressor.add(Dropout(0.25))

regressor.add(Flatten())

regressor.add(Dense(1))

regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(x_train, y_train, epochs=100, batch_size=64)

# prepare test data for prediction

inputs = data[len(data)-len(close_test)-timestep:]
inputs = inputs.values.reshape(-1,1)
inputs = scaler.transform(inputs)
x_test = []
for i in range(timestep,inputs.shape[0]):
    x_test.append(inputs[i-timestep:i,0])
x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
predicted_data = regressor.predict(x_test)
predicted_data = scaler.inverse_transform(predicted_data)

# plot results
data_test = np.array(close_test)
data_test = data_test.reshape(len(data_test), 1)
plt.figure(figsize=(8,4), dpi=80, facecolor='w', edgecolor='k')
plt.plot(data_test, color='r', label='true-result')
plt.plot(predicted_data, color='b', label='predicted result')
plt.legend()
plt.xlabel('Time(50 days)')
plt.ylabel("Close Values")
plt.grid(True)
plt.show()


# LSTM

model = Sequential()

model.add(LSTM(10,input_shape=(None,1), activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer="adam")
model.fit(x_train, y_train, epochs=100, batch_size=32)



inputs = data[len(data)-len(close_test)-timestep:]
inputs=inputs.values.reshape(-1,1)
inputs=scaler.transform(inputs)
x_test = []
for i in range(timestep, inputs.shape[0]):
    x_test.append(inputs[i-timestep:i,0])
x_test=np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)
predicted_data=model.predict(x_test)
predicted_data=scaler.inverse_transform(predicted_data)

# plot results

data_test=np.array(close_test)
data_test = data_test.reshape(len(data_test), 1)
plt.figure(figsize=(8,4), dpi=80, facecolor='w', edgecolor='k')
plt.plot(data_test,color='r',label='true_result')
plt.plot(predicted_data,color='b',label='predicted result')
plt.legend()
plt.xlabel('Time(50 days)')
plt.ylabel("Close Values")
plt.grid(True)
plt.show()