import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from pandas import concat
from pandas import Series
from math import sqrt


def make_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
def invert_diff(history, yhat, interval=1):
	return yhat + history[-interval]

def scale(train, test):
	scale_f = MinMaxScaler(feature_range=(-1,1))
	scale_f = scale_f.fit(train)
	train = train.reshape(train.shape[0], train.shape[1])
	train_s = scale_f.transform(train)
	test = test.reshape(test.shape[0], test.shape[1])
	test_s = scale_f.transform(test)
	return scale_f, train_s, test_s

def invert_scale(scale_f, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scale_f.inverse_transform(array)
	return inverted[0, -1]

def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:,-1]
	X = X.reshape(X.shape[0],1,X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam') 
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

def lstm_forecast(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

df = pd.read_csv('data_files/WIKI-AAPL.csv',usecols=[11],engine='python', skipfooter=3)
df = df['Adj. Close']
df = df.astype('float32')

X = df.values
diff_vals = difference(X,2)

supervised = make_supervised(diff_vals, 1)
supervised_vals = supervised.values

train_len = int(len(supervised_vals)*0.80)
test_len = len(supervised_vals) - train_len
train, test = supervised_vals[0:train_len,:],supervised_vals[train_len:len(supervised_vals),:]

scaler, trains_s, test_s = scale(train,test)
lstm_model = fit_lstm(trains_s, 1, 1, 4)
train_r = trains_s[:, 0].reshape(len(trains_s),1, 1)

lstm_model.predict(train_r, batch_size=1)

predictionlist = list()
for i in range(len(test_s)):
	Z, y = test_s[i, 0:-1], test_s[i, -1]
	yhat = lstm_forecast(lstm_model,1,Z)
	yhat = invert_scale(scaler, Z, yhat)
	yhat = invert_diff(X, yhat, len(test_s)+1-i)
	predictionlist.append(yhat)
	expected = X[len(train)+i+1]
	print('Trading day=%d, Predicted=%f, Expected=%f'%(i+1, yhat, expected))


rmse = sqrt(mean_squared_error(X[-test_len:], predictionlist))
print('Test RMSE: %.3f' % rmse)
plt.plot(X[-test_len:])
plt.ion()
plt.pause(10)
plt.plot(predictionlist)
plt.pause(10)





