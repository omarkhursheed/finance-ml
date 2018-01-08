from __future__ import print_function

from pandas import Series
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


series = Series.from_csv('LSE-ABDP-dataset.csv')
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
#bias = 165.904728
months_in_year = 12
diff = difference(history, months_in_year)
# predict
model = ARIMA(diff, order=(2,0,1))
	
for i in range(len(test)):
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))

validation_series = Series.from_csv('LSE-ABDP-validation.csv')
Y = validation_series.values
Y = Y.astype('float32')
validation_predictions = list()
history_validation = [x for x in validation_series]
error_list = []
for i in range(len(validation_series)):
	model_fit = model.fit(trend='nc', disp=0)
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history_validation, yhat, months_in_year)
	validation_predictions.append(yhat)
	obs = validation_series[i]
	history_validation.append(obs)
	print('>Predicted=%.3f, Expected=%3.f'  % (yhat, obs))
	error_list.append(abs((yhat - obs)*100/obs))
