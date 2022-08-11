from sklearn.preprocessing import MinMaxScaler
from numpy import asarray
import pandas as pd



data = pd.read_csv('..\data\CPI_data.csv')
data['Dates'] = pd.to_datetime(data['Dates'])
daat = data.sort_values("Dates",inplace=True)

# scale all the data between 0 and 1 
scaler = MinMaxScaler()
scaled_CPI = asarray(data['CPIGR']).reshape(-1,1)
scaled_CPI = scaler.fit_transform(scaled_CPI)

p = 20
#print(scaled_CPI.shape)

# We omit the last 20 observations for our out of sample forecast
out_of_sample_forecast_input = scaled_CPI[960:,0]

# Retain all the data minus the last 20 observatinos for forecasting
scaled_CPI = scaled_CPI[:960,0]
print(scaled_CPI)