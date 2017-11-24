from flask import Flask, render_template, redirect, url_for, request
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.svm import SVR
from flask_table import Table, Col
from os import listdir
from os.path import isfile, join
import os, os.path
import numpy as np
import pandas as pd
import time, datetime
import subprocess
import pickle
import math
import csv

import matplotlib.pyplot as plt 
import datetime
from matplotlib import style
import tkinter


app = Flask(__name__)

dname = os.path.dirname(os.path.abspath(__file__))

onlyfiles = [f for f in listdir(dname+'/data_files/') if isfile(join(dname+'/data_files/', f))]
codes_list_file = pd.read_csv(dname+'/WIKI-datasets-codes.csv',names=["Code","Name"])

onlyfiles.sort()

class ItemTable(Table):
    #classes = ['table','table-striped','table-bordered','table-hover']
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

@app.route('/')
def main():
	return render_template('home.html')

@app.route('/stockselect')
def stockselect():
	onlyfiles = [f for f in listdir(dname+'/data_files/') if isfile(join(dname+'/data_files/', f))]
	onlyfiles.sort()
	return render_template('stockselect.html', onlyfiles=onlyfiles)
	
@app.route('/result',methods = ['POST','GET'])
def result():
	if request.method == 'POST':
		selected_stock = request.form['file_select']
		selected_stock = selected_stock.replace('/','-',1)

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
		
		model_svr = pickle.load(open(join(dname+'/models/svr_fit/', selected_stock+'svr.sav'),'rb'))
		model_lr = pickle.load(open(join(dname+'/models/lr_fit/', selected_stock+'lr.sav'),'rb'))
		model_mlp = pickle.load(open(join(dname+'/models/mlp_fit/', selected_stock+'mlp.sav'),'rb'))
		model_svru = pickle.load(open(join(dname+'/models/svr_unfit/', selected_stock+'svr.sav'),'rb'))
		model_lru = pickle.load(open(join(dname+'/models/lr_unfit/', selected_stock+'lr.sav'),'rb'))
		model_mlpu = pickle.load(open(join(dname+'/models/mlp_unfit/', selected_stock+'mlp.sav'),'rb'))

		conf1 = model_svr.score(X_test, y_test)
		temp1 = str(conf1*100.0)
		conf2 = model_lr.score(X_test, y_test)
		temp2 = str(conf2*100.0)
		conf3 = model_mlp.score(X_test, y_test)
		temp3 = str(conf3*100.0)
		
		seed = 7
		num_samples = 10
		test_size = 0.33
		#kfold = model_selection.KFold(n_splits=5, random_state=seed)
		kfold = model_selection.ShuffleSplit(n_splits=5, test_size=test_size, random_state=seed)
		confsvr = model_selection.cross_val_score(model_svru, X, y, cv=kfold)
		temp1c = str(confsvr.mean()*100.0)
		conflr = model_selection.cross_val_score(model_lru, X, y, cv=kfold)
		temp2c = str(conflr.mean()*100.0)
		confmlp = model_selection.cross_val_score(model_mlpu, X, y, cv=kfold)
		temp3c = str(confmlp.mean()*100.0)

		return render_template("result.html",result1 = temp1,result2 = temp2,result3 = temp3,result1c = temp1c,result2c = temp2c,result3c = temp3c)

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
		svr = SVR()
		pickle.dump(svr,open(join(dname+'/models/svr_unfit/', selected_stock+'svr.sav'),'wb'))
		svr.fit(X_train, y_train)
		
		lr = LinearRegression()
		pickle.dump(lr,open(join(dname+'/models/lr_unfit/', selected_stock+'lr.sav'),'wb'))
		lr.fit(X_train, y_train)
		
		mlp = MLPRegressor()
		pickle.dump(mlp,open(join(dname+'/models/mlp_unfit/', selected_stock+'mlp.sav'),'wb'))
		mlp.fit(X_train, y_train)

		pickle.dump(svr,open(join(dname+'/models/svr_fit/', selected_stock+'svr.sav'),'wb'))
		pickle.dump(lr,open(join(dname+'/models/lr_fit/', selected_stock+'lr.sav'),'wb'))
		pickle.dump(mlp,open(join(dname+'/models/mlp_fit/', selected_stock+'mlp.sav'),'wb'))
		
		return adminsec()

@app.route('/table',methods = ['POST','GET'])
def table():
	os.chdir(dname)
	myTable = pickle.load(open('my.pkl','rb'))
	table = ItemTable(myTable, border=True)

	return render_template('table.html',table=table)

@app.route('/createtable',methods = ['POST','GET'])
def createtable():
	if request.method == 'POST':
		subprocess.call(['gnome-terminal', '-e', 'python3 '+dname+'/createtable.py'])
		
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
	onlyfiles = [f for f in listdir(dname+'/data_files/') if isfile(join(dname+'/data_files/', f))]
	onlyfiles.sort()
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
