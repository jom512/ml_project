import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("bitcoin.csv")

predict_days = 30

data_train= data[:len(data)-predict_days]
data_test= data[len(data)-predict_days:]

training_set = data_train.values
min_max_scaler = MinMaxScaler()
training_set = min_max_scaler.fit_transform(training_set)

x_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
x_train = np.reshape(x_train, (len(x_train), 1, 1))

batch_size = 10
num_epochs = 60

sgd = optimizers.SGD(lr=0.01, clipnorm=0.5)

rnn = Sequential()
rnn.add(LSTM(units = 4, activation = 'sigmoid', input_shape=(None, 1)))
rnn.add(Dense(units = 1))
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
rnn.fit(x_train, y_train, batch_size = batch_size, epochs = num_epochs)

test_set = data_test.values

inputs = np.reshape(test_set, (len(test_set), 1))
inputs = min_max_scaler.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))

predicted_price = rnn.predict(inputs)
predicted_price = min_max_scaler.inverse_transform(predicted_price)

red = np.append(data_train.values, data_test.values)
blue = np.append(data_train.values, predicted_price[:])

plt.plot(blue, color = 'blue', label = 'Predicted')
plt.plot(red, color='red', label='Real')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
