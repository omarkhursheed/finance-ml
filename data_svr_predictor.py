from pandas import DataFrame
from pandas import concat
import pandas as pd
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	agg = concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg
df = pd.read_csv("data_files/LSE-ABDP.csv")
df = df[['Date','Price']]

data = df['Price'].values.tolist()
dates = df['Date'].values.tolist()
dates = dates[1:]
print(len(data))
print(len(dates))
se = pd.Series(dates)
new_df = series_to_supervised(data)
print(new_df.shape)
new_df['Date'] = se.values
new_df = new_df.set_index(['Date'])

print(new_df)
