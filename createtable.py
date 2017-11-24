from flask_table import Table, Col
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.svm import SVR
from os import listdir
from os.path import isfile, join
import os, os.path
import numpy as np
import pandas as pd
import time, datetime
import subprocess
import pickle
import math

import matplotlib.pyplot as plt 
import datetime
from matplotlib import style
import tkinter

dname = os.path.dirname(os.path.abspath(__file__))

onlyfiles = [f for f in listdir(dname+'/data_files/') if isfile(join(dname+'/data_files/', f))]
onlyfiles.sort()

# Declare your table
class ItemTable(Table):
    name = Col('Name')
    svr = Col('SVR')
    lr = Col('LR')
    mlp = Col('MLP')
    last = Col('Last')
    nex = Col('Predicted')
    change = Col('Change')

# Get some objects
class Item(object):
    def __init__(self, name, svr, lr, mlp, last, nex, change):
        self.name = name
        self.svr = svr
        self.lr = lr
        self.mlp = mlp
        self.last = last
        self.nex = nex
        self.change = change

def createtable():
	os.chdir(dname)

	items = []

	for x in onlyfiles:
		df = pd.read_csv(os.path.join('data_files',x))
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
	
		df.dropna(inplace=True)
		y = np.array(df['label'])
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
		
		loaded_model_svr = pickle.load(open(join(dname+'/models/svr_unfit/', x+'svr.sav'),'rb'))
		loaded_model_lr = pickle.load(open(join(dname+'/models/lr_unfit/', x+'lr.sav'),'rb'))
		loaded_model_mlp = pickle.load(open(join(dname+'/models/mlp_unfit/', x+'mlp.sav'),'rb'))

		num_instances = len(X)

		seed = 7
		num_samples = 10
		test_size = 0.33
		#kfold = model_selection.KFold(n_splits=5, random_state=seed)
		kfold = model_selection.ShuffleSplit(n_splits=5, test_size=test_size, random_state=seed)

		X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

		confsvr = model_selection.cross_val_score(loaded_model_svr, X, y, cv=kfold)
		temp1 = str("%.3f%%" % (confsvr.mean()*100.0))
		conflr = model_selection.cross_val_score(loaded_model_lr, X, y, cv=kfold)
		temp2 = str("%.3f%%" % (conflr.mean()*100.0))
		confmlp = model_selection.cross_val_score(loaded_model_mlp, X, y, cv=kfold)
		temp3 = str("%.3f%%" % (confmlp.mean()*100.0))

		loaded_model_lr.fit(X_train,y_train)
		forecast_set = loaded_model_lr.predict(X_lately)
		df['Forecast'] = np.nan
		last_date = df.iloc[-1].name
		last_unix = last_date.timestamp()
		one_day = 86400
		next_unix = last_unix + one_day

		for i in forecast_set:
			next_date = datetime.datetime.fromtimestamp(next_unix)
			next_unix += 86400
			df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

		last_for_compare = df['Forecast'].iloc[-1]
		change = str("%.3f" % (last_for_compare - first_for_compare))
		lastx = str("%.3f" % last_for_compare)


		items.append(dict(name=x,svr=temp1,lr=temp2,mlp=temp3,last=first_for_compare, nex=lastx, change=change))

	pickle.dump(items,open('my.pkl','wb'))

createtable()
