from __future__ import print_function

import datetime
import pandas as pd 
import numpy as np
import sklearn

from pandas_datareader import DataReader
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix
#from sklearn.qda import QDA
from sklearn.svm import LinearSVC, SVC

def create_serice(symbol, start, end, lags = 5):
	ts = DataReader(
		symbol, "yahoo",start-datetime.timedelta(days = 365),
		end)

	tslag = pd.DataFrame(index = ts.index)
	tslag["Today"] = ts["Adj Close"]
	tslag["Volume"] = ts["Volume"]
	tsret["Today"] = tslag["Today"].pct_change()*100.0

	for i,x in enumerate(tsret["Today"]):
		if(abs(x) < 0.0001):
			tsret["Today"][i] = 0.0001

	for i in range(0, lags):
		tsret["Lag%s" % str(i+1)] = \
		tslag["Lag%s" % str(i+1)].pct_change()*100.0

	tsret["Direction"] = np.sign(tsret["Today"])
	tsret = tsret[tsret.index >= start_dae]

	return tsret


if __name__ == "__main__":
	snpret = create_serice("^GSPC",datetime.datetime(2001,1,10),datetime.datetime(2005,12,31), lags=5)
	X = snpret[["Lag1","Lag2"]]
	y = snpret["Direction"]

	start_test = datetime.datetime(2005, 1,1)
	X_train = X[X.index < start_test]
	X_test = X[X.index >= start_test]
	y_train = y[y.index < start_test]
	y_test = y[y.index() >= start_test]

	print("Hit rates")
	models = [("LR, LogisticRegression()"),
	("LSV", LinearSVC()),
	("RSVM", SVC(
	C=1000000.0, cache_size=200, class_weight=None,
	coef0=0.0, degree=3, gamma=0.0001,
	max_iter=-1, probability=False, random_state=None,
	shrinking=True, tol=0.001, verbose=False)
	),
	]

	for m in models:
		m[1].fit(X_train, y_train)
		pred = m[1].predit(X_test)

		print("%s:\n0.3f"%(m[0],m[1].score(X_test, y_test)))