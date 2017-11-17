from flask import Flask, render_template, redirect, url_for, request
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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


app = Flask(__name__)

onlyfiles = [f for f in listdir(os.path.dirname(os.path.realpath(__file__))+'/data_files/') if isfile(join(os.path.dirname(os.path.realpath(__file__))+'/data_files/', f))]

dname = os.path.dirname(os.path.abspath(__file__))
onlyfiles.sort()

@app.route('/')
def main():
	return render_template('home.html')

@app.route('/stockselect')
def stockselect():
	return render_template('stockselect.html', files=onlyfiles)
	
@app.route('/result',methods = ['POST','GET'])
def result():
	if request.method == 'POST':
		selected_stock = request.form['file_select']
		
		os.chdir(dname)
		
		df = pd.read_csv(os.path.join('data_files',selected_stock))
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
		
		loaded_model_svr = pickle.load(open(join(dname+'/models/', selected_stock+'svr.sav'),'rb'))
		loaded_model_lr = pickle.load(open(join(dname+'/models/', selected_stock+'lr.sav'),'rb'))
		loaded_model_mlp = pickle.load(open(join(dname+'/models/', selected_stock+'mlp.sav'),'rb'))

		confidence1 = loaded_model_svr.score(X_test, y_test)
		temp1 = str(confidence1)
		confidence2 = loaded_model_lr.score(X_test, y_test)
		temp2 = str(confidence2)
		confidence3 = loaded_model_mlp.score(X_test, y_test)
		temp3 = str(confidence3)
		

		return render_template("result.html",result1 = temp1,result2 = temp2,result3 = temp3)

@app.route('/train',methods = ['POST','GET'])
def train():
	if request.method == 'POST':
		selected_stock = request.form['file_select']
		os.chdir(dname)
		
		df = pd.read_csv(os.path.join('data_files',selected_stock))

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

		pickle.dump(clf,open(join(dname+'/models/', selected_stock+'svr.sav'),'wb'))
		pickle.dump(lr,open(join(dname+'/models/', selected_stock+'lr.sav'),'wb'))
		pickle.dump(mlp,open(join(dname+'/models/', selected_stock+'mlp.sav'),'wb'))
		
		return adminsec()

@app.route('/trainall',methods = ['POST','GET'])
def trainall():
	if request.method == 'POST':
		subprocess.call(['gnome-terminal', '-e', 'python3 '+dname+'/trainall.py'])
		
		return adminsec()

@app.route('/updateprices',methods = ['POST','GET'])
def updateprices():
	if request.method == 'POST':
		subprocess.call(['gnome-terminal', '-e', 'python3 '+dname+'/data_extractor.py'])
		
		return adminsec()

@app.route('/adminsec')
def adminsec():
	return render_template('adminsec.html', files=onlyfiles)

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

if __name__ == '__main__':
	app.run(debug = True)
