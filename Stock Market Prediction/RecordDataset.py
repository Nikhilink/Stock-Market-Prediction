import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time
import pandas_datareader as dr
import nsepy as nse
from datetime import date

data = nse.get_history('NIFTY 50',start=date(2020,1,3),end=date(2020,7,4),index=True)
print(data)
data.to_csv("NiftyChaAajChaData.csv")

#api_key = '7CHHSNPAW1MAOF74'

print("Expired API")
