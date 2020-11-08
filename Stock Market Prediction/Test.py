import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import nsepy as nse
from datetime import date
import matplotlib.pyplot as plt 

dataset = pd.read_csv("datasets/NIFTY 50.csv",index_col="Date",parse_dates=True)

scaler = MinMaxScaler(feature_range=(0,1))
inputs = scaler.fit_transform(dataset)

data = nse.get_history('NIFTY 50',start=date(2020,4,11),end=date(2020,11,4),index=True)
dataset_test = data
dataset_test = dataset_test.head(20)

real_stock_price = dataset_test.iloc[:,1:2].values

test_set = dataset_test['Open']
test_set = pd.DataFrame(test_set)

regressor = load_model("NiftyModel.h5")

dataset_total = pd.concat((dataset['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
pdstock = regressor.predict(X_test)
pdstock = scaler.inverse_transform(pdstock)

plt.plot(real_stock_price,color='red',label='Real Nifty Stock Price')
plt.plot(pdstock,color='blue',label='Predicted Nifty Stock Price')
plt.title("Nifty Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Nifty Stock Price')
plt.legend()
plt.show()