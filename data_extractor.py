import quandl
import pandas as pd
import json
import string
import os, os.path
quandl.ApiConfig.api_key = 'yvR18HYhyiyR51fL9UM5'
dname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dname)

codes_list_file = pd.read_csv("WIKI-datasets-codes.csv",names=["Code","Name"])

for exchange in codes_list_file['Code']:
	df = quandl.get(exchange)
	exchange1 = exchange.replace('/','-',1)
	file_path =  'data_files/' + exchange1 + '.csv' 
	
	df.to_csv(file_path)
	print(exchange+" - pulled")
