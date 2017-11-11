import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import math
import matplotlib.pyplot as plt 
import datetime
from matplotlib import style
import tkinter
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('data_files/WIKI-AAPL.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
#preprocessing the data
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
#measure of volatility
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf1 = SVR()
clf1.fit(X_train, y_train)
confidence1 = clf1.score(X_test, y_test)
print('SVR : '+str(confidence1))

clf2 = MLPRegressor()
clf2.fit(X_train, y_train)
confidence2 = clf2.score(X_test, y_test)
print('MLP : '+str(confidence2))

clf = LinearRegression()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('LR : '+str(confidence))
#Add forecasting code for submission on 11th November, 2017

forecast_set = clf.predict(X_lately)
#print(forecast_set, confidence, forecast_out)
df['Forecast'] = np.nan
#print(df.iloc[-1])
last_date = df.iloc[-1].name
print(last_date)
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += 86400
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

#print( df.tail())
df['Adj. Close'].plot()
df['Forecast'].plot()

plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
