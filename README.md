# Forecasting-Airlines-data-
Forecast the Airlines Passengers data set. Prepare a document for each model explaining  how many dummy variables you have created and RMSE value for each model. Finally which model you will use for  Forecasting.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load the dataset from Excel
data = pd.read_excel('Airlines+Data.xlsx')

# Preprocess the data
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
data['Passengers'] = pd.to_numeric(data['Passengers'], errors='coerce')

# Exploratory Data Analysis (EDA)
# Time series plot
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Passengers'], label='Actual', color='blue')
plt.title('Monthly Passenger Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.grid(True)
plt.show()

# Check for stationarity using Dickey-Fuller test
def adf_test(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

adf_test(data['Passengers'])
from statsmodels.tsa.seasonal import seasonal_decompose
# Decompose the time series
decomposition = seasonal_decompose(data['Passengers'], model='additive')
plt.figure(figsize=(12, 8))
decomposition.plot()
plt.title('Decomposition Plot')
plt.show()

# Autocorrelation plot
plt.figure(figsize=(12, 6))
plot_acf(data['Passengers'], lags=50)
plt.title('Autocorrelation Plot')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

# Partial Autocorrelation plot
plt.figure(figsize=(12, 6))
plot_pacf(data['Passengers'], lags=24)
plt.title('Partial Autocorrelation Plot')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()

# Seasonal Decomposition
plt.figure(figsize=(12, 8))
decomposition = seasonal_decompose(data['Passengers'], model='multiplicative')
decomposition.plot()
plt.title('Seasonal Decomposition (Multiplicative)')
plt.show()

# Model Building and Forecasting
# Split the data
train_data = data.iloc[:-12]
test_data = data.iloc[-12:]

# Build and evaluate models
best_rmse = np.inf
best_model = None

for p in range(0, 3):
    for d in range(0, 2):
        for q in range(0, 3):
            for P in range(0, 2):
                for D in range(0, 2):
                    for Q in range(0, 2):
                        order = (p, d, q)
                        seasonal_order = (P, D, Q, 12)
                        model = SARIMAX(train_data['Passengers'], order=order, seasonal_order=seasonal_order)
                        fitted_model = model.fit()
                        forecast = fitted_model.get_forecast(steps=len(test_data))
                        predicted = forecast.predicted_mean
                        rmse = np.sqrt(mean_squared_error(test_data['Passengers'], predicted))

                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_model = fitted_model

# Select the best model
final_model = best_model

# Forecasting
forecast = final_model.get_forecast(steps=12)
forecasted_passengers = forecast.predicted_mean

# RMSE values for each model can be accessed through the best_rmse variable

# Print the forecasted passengers and RMSE values
print("Forecasted Passengers:")
print(forecasted_passengers)
print("\nRMSE Values:")
print(best_rmse)

# Visualizations (cont'd)
# Time series plot with forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Passengers'], label='Actual', color='blue')
plt.plot(test_data.index, forecasted_passengers, label='Forecast', color='red')
plt.title('Passenger Forecasting with SARIMA')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.grid(True)
plt.show()
