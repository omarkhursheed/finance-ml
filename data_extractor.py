import quandl
import pandas as pd
import json
import string
quandl.ApiConfig.api_key = 'yvR18HYhyiyR51fL9UM5'
#data  = quandl.get("NSE/BSLGOLDETF")
#data = quandl.get("https://www.quandl.com/api/v3/databases/LSE/codes.json")
#print(data)
codes_list_file = pd.read_csv("LSE-datasets-codes.csv")

for exchange in codes_list_file['LSE/DAILY_TRADES']:
	df = quandl.get(exchange)
	exchange1 = exchange.replace('/','-',1)
	file_path =  'data_files/' + exchange1 + '.csv' 
	
	df.to_csv(file_path)
	print("all")