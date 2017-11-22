import os, os.path
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import math
import pickle

dname = os.path.dirname(os.path.abspath(__file__))

onlyfiles = [f for f in listdir(os.path.dirname(os.path.realpath(__file__))+'/data_files/') if isfile(join(os.path.dirname(os.path.realpath(__file__))+'/data_files/', f))]
onlyfiles.sort()

def train():
	os.chdir(dname)
	for x in onlyfiles:
		df = pd.read_csv(os.path.join('data_files',x))
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
		
		clf = SVR()
		clf.fit(X_train, y_train)
		
		lr = LinearRegression()
		lr.fit(X_train, y_train)
		
		mlp = MLPRegressor()
		mlp.fit(X_train, y_train)

		pickle.dump(clf,open(join(dname+'/models/svr_fit/', x+'svr.sav'),'wb'))
		pickle.dump(lr,open(join(dname+'/models/lr_fit/', x+'lr.sav'),'wb'))
		pickle.dump(mlp,open(join(dname+'/models/mlp_fit/', x+'mlp.sav'),'wb')) 

		print(x+" - trained")
		
train() 
