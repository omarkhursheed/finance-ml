import quandl
import pandas as pd
#api_key = yvR18HYhyiyR51fL9UM5
data  = quandl.bulkdownload("ZEA?api_key=yvR18HYhyiyR51fL9UM5")

print(data)