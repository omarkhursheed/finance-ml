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

print(df)