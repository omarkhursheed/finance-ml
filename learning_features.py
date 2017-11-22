import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import matplotlib
matplotlib.use('TkAgg')
import math
import matplotlib.pyplot as plt 
import datetime
from matplotlib import style
import tkinter
import warnings
import os, os.path

dname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dname)

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
first_for_compare = df['Adj. Close'].iloc[-1]

print(df.tail())
df.dropna(inplace=True)
y = np.array(df['label'])
print(first_for_compare)

num_instances = len(X)

seed = 7
num_samples = 10
test_size = 0.33
#kfold = model_selection.KFold(n_splits=5, random_state=seed)
kfold = model_selection.ShuffleSplit(n_splits=5, test_size=test_size, random_state=seed)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf1 = SVR()

confidence1kfold = model_selection.cross_val_score(clf1, X, y, cv=kfold)
print("SVR (KFold) : %.3f%% (%.3f%%)" % (confidence1kfold.mean()*100.0, confidence1kfold.std()*100.0))

clf1.fit(X_train, y_train)
confidence1 = clf1.score(X_test, y_test)
print("SVR : %.3f%%" % (confidence1*100.0))

clf2 = MLPRegressor()

confidence2kfold = model_selection.cross_val_score(clf2, X, y, cv=kfold)
print("MLP (KFold) : %.3f%% (%.3f%%)" % (confidence2kfold.mean()*100.0, confidence2kfold.std()*100.0))

clf2.fit(X_train, y_train)
confidence2 = clf2.score(X_test, y_test)
print("MLP : %.3f%%" % (confidence2*100.0))

clf = LinearRegression()

confidencekfold = model_selection.cross_val_score(clf, X, y, cv=kfold)
print("LR (KFold) : %.3f%% (%.3f%%)" % (confidencekfold.mean()*100.0, confidencekfold.std()*100.0))

clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("LR : %.3f%%" % (confidence*100.0))

#Add forecasting code for submission on 11th November, 2017
forecast_set = clf1.predict(X_lately)
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
last_for_compare = df['Forecast'].iloc[-1]
df['Adj. Close'].plot()
df['Forecast'].plot()

plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()