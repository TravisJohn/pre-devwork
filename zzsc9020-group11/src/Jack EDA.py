# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir('D:/Uni/Master of Data Science/ZZSC9020/zzsc9020-group11/Project code')

# Load the datasets
temperature_file = 'NSW Data/temperature_nsw.csv'
demand_file = 'NSW Data/totaldemand_nsw.csv'
public_holidays = 'NSW Data/Aus_public_hols_2009-2022-1.csv'
humidity_solar1 = 'NSW Data/humiditysolar1.csv'
humidity_solar2 = 'NSW Data/humiditysolar2.csv'

# Read in demand and temperature data
temperature_data = pd.read_csv(temperature_file)
demand_data = pd.read_csv(demand_file)
pub_holiday = pd.read_csv(public_holidays)

# Read in solar and humidity data
humiditysolar1 = pd.read_csv(humidity_solar1)
humiditysolar2 = pd.read_csv(humidity_solar2)
humidity_solar_combined = pd.concat([humiditysolar1, humiditysolar2], axis=0)
humidity_solar_combined['Date'] = pd.to_datetime(humidity_solar_combined['Date'], dayfirst=True, errors='coerce')

# Replace '24:00' with '00:00' in the 'Time' column
humidity_solar_combined.loc[humidity_solar_combined['Time'] == '24:00', 'Time'] = '00:00'
# Ensure Date is in datetime format and shift the Date by 1 day for rows where Time is '00:00'
humidity_solar_combined['Date'] = pd.to_datetime(humidity_solar_combined['Date'], errors='coerce')
# Shift Date by 1 day where Time is '00:00'
humidity_solar_combined.loc[humidity_solar_combined['Time'] == '00:00', 'Date'] = humidity_solar_combined.loc[humidity_solar_combined['Time'] == '00:00', 'Date'] + pd.Timedelta(days=1)
humidity_solar_combined['DATETIME'] = pd.to_datetime(humidity_solar_combined['Date'].astype(str) + ' ' + humidity_solar_combined['Time'])
humidity_solar_combined = humidity_solar_combined.drop(columns=['Date', 'Time'])

# Rename the columns
humiditysolar_data = humidity_solar_combined.rename(columns={'CHULLORA HUMID 1h average [%]': 'Humidity', 'CHULLORA SOLAR 1h average [W/m≤]': 'SolarRadiation'})
humiditysolar_data.dropna(inplace=True)



# Convert DATETIME columns to datetime type for both datasets and merge them on 'DATETIME'
temperature_data['DATETIME'] = pd.to_datetime(temperature_data['DATETIME'], format='%d/%m/%Y %H:%M')
demand_data['DATETIME'] = pd.to_datetime(demand_data['DATETIME'], format='%d/%m/%Y %H:%M')
pub_holiday['Date'] = pd.to_datetime(pub_holiday['Date'], format = '%Y-%m-%d').dt.date

# Merge datasets on 'DATETIME'
merged_data = pd.merge(demand_data, temperature_data, on='DATETIME', how='inner')
merged_data = pd.merge(merged_data, humiditysolar_data, on='DATETIME', how='left')

# Extract temporal features (hour, day of the week, month)
merged_data['Hour'] = merged_data['DATETIME'].dt.hour
merged_data['DayOfWeek'] = merged_data['DATETIME'].dt.dayofweek
merged_data['Month'] = merged_data['DATETIME'].dt.month
merged_data['Date'] = merged_data['DATETIME'].dt.date
merged_data['Year'] = merged_data['DATETIME'].dt.year

# Create a clean dataset by ensuring numeric values for 'TOTALDEMAND' and 'TEMPERATURE'
merged_data['TOTALDEMAND'] = pd.to_numeric(merged_data['TOTALDEMAND'], errors='coerce')
merged_data['TEMPERATURE'] = pd.to_numeric(merged_data['TEMPERATURE'], errors='coerce')
cleaned_data = merged_data.dropna(subset=['TOTALDEMAND', 'TEMPERATURE'])

# Convert the 'DATETIME' column to ensure it is in datetime format correctly
cleaned_data['DATETIME'] = pd.to_datetime(cleaned_data['DATETIME'], errors='coerce')

# Create DateOnly column for easier matching with public holidays
cleaned_data['DateOnly'] = cleaned_data['DATETIME'].dt.date

pub_holiday['IsHoliday'] = 1
cleaned_data['REGIONID'] = 'NSW'
cleaned_data = pd.merge(cleaned_data, pub_holiday[['Date', 'State', 'IsHoliday']], 
                        left_on=('Date', 'REGIONID'), right_on=('Date', 'State'), how = 'left')

cleaned_data['IsHoliday'].fillna(0, inplace = True)
cleaned_data.drop(columns = ["State"], inplace = True)

# Update 'DayType' to include Public Holiday
cleaned_data['DayType'] = cleaned_data.apply(lambda row: 'Public Holiday' if row['IsHoliday'] else (
    'Weekend' if row['DayOfWeek'] >= 5 else 'Weekday'), axis=1)

# 1. Density plots of total demand and temperature
sns.set_palette(sns.color_palette("muted"))
# Histogram for 'TOTALDEMAND' with a density line
plt.figure(figsize=(10,6))
sns.histplot(cleaned_data['TOTALDEMAND'], kde=True, color='blue', bins=50)
plt.title('Histogram of Total Demand with Density Line')
plt.xlabel('Total Demand (MW)')
plt.ylabel('Frequency')
plt.show()

# Histogram for 'TEMPERATURE' with a density line
plt.figure(figsize=(10,6))
sns.histplot(cleaned_data['TEMPERATURE'], kde=True, color='orange', bins=50)
plt.title('Histogram of Temperature with Density Line')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.show()


# 2. Line chart of total demand with month on x-axis and grouped by year
grouped_by_month_year = cleaned_data.groupby(['Year', 'Month'])['TOTALDEMAND'].mean().reset_index()

plt.figure(figsize=(12,6))
for year in grouped_by_month_year['Year'].unique():
    yearly_data = grouped_by_month_year[grouped_by_month_year['Year'] == year]
    plt.plot(yearly_data['Month'], yearly_data['TOTALDEMAND'], label=f'Year {year}')

plt.title('Average Total Demand by Month Grouped by Year')
plt.xlabel('Month')
plt.ylabel('Average Total Demand (MW)')
plt.legend()
plt.show()

# 3. Box plot of total demand grouped by public holidays, weekends, and weekdays
plt.figure(figsize=(10,6))
sns.boxplot(x='DayType', y='TOTALDEMAND', data=cleaned_data, order=['Public Holiday', 'Weekend', 'Weekday'])
plt.title('Total Demand Grouped by Public Holiday, Weekend, and Weekdays')
plt.xlabel('Day Type')
plt.ylabel('Total Demand (MW)')
plt.show()

# 4. Scatter plot of total demand and temperature, color by month
plt.figure(figsize=(10,6))
scatter = plt.scatter(x=cleaned_data['TEMPERATURE'], y=cleaned_data['TOTALDEMAND'], 
                      c=cleaned_data['Month'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Month')
plt.title('Scatter Plot of Total Demand vs Temperature, Colored by Month')
plt.xlabel('Temperature (°C)')
plt.ylabel('Total Demand (MW)')
plt.show()

# 5. Line chart of total demand with x-axis as hour and grouped by month (averaged across years)
grouped_by_hour_month = cleaned_data.groupby(['Month', 'Hour'])['TOTALDEMAND'].mean().reset_index()

plt.figure(figsize=(12,6))
for month in grouped_by_hour_month['Month'].unique():
    monthly_data = grouped_by_hour_month[grouped_by_hour_month['Month'] == month]
    plt.plot(monthly_data['Hour'], monthly_data['TOTALDEMAND'], label=f'Month {month}')

plt.title('Average Total Demand by Hour, Grouped by Month')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Total Demand (MW)')
plt.legend()
plt.show()


# Box plot for electricity demand by hour of the day
plt.figure(figsize=(10,6))
sns.boxplot(x='Hour', y='TOTALDEMAND', data=cleaned_data)
plt.title('Electricity Demand by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Demand (MW)')
plt.show()

# Box plot for electricity demand by day of the week
plt.figure(figsize=(10,6))
sns.boxplot(x='DayOfWeek', y='TOTALDEMAND', data=cleaned_data)
plt.title('Electricity Demand by Day of the Week')
plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
plt.ylabel('Total Demand (MW)')
plt.show()

# Perform PACF on electricity demand and temperature

from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

# Plot the PACF for electricity demand
plt.figure(figsize=(10,6))
plot_pacf(cleaned_data['TOTALDEMAND'], lags=60, method='ywm')  # lags=60 for the PACF up to 30 hours (30-min intervals)
plt.title('Partial Autocorrelation of Electricity Demand')
plt.xlabel('Lag')  # X-axis title
plt.ylabel('PACF')  # Y-axis title
plt.show()


# Plot the PACF for Temp
plt.figure(figsize=(10,6))
plot_pacf(cleaned_data['TEMPERATURE'], lags=60, method='ywm')  # lags=48 because 48*30min = 1 day
plt.title('Partial Autocorrelation of Temperature')
plt.show()

# Granger Causality Test
from statsmodels.tsa.stattools import grangercausalitytests
# Combine temperature and electricity demand in one DataFrame
granger_data = cleaned_data[['TOTALDEMAND', 'TEMPERATURE']]

# Perform Granger causality test to check if lagged temperature causes electricity demand
grangercausalitytests(granger_data, maxlag= 8)

from statsmodels.graphics.tsaplots import plot_acf

# Plot the ACF for electricity demand
plt.figure(figsize=(10,6))
plot_acf(cleaned_data['TOTALDEMAND'], lags=48)  # lags=48 to see patterns up to 24 hours
plt.title('Autocorrelation of Electricity Demand')
plt.show()

# from statsmodels.tsa.stattools import ccf

# # Calculate the cross-correlation between temperature and demand
# cross_corr = ccf(cleaned_data['TEMPERATURE'], cleaned_data['TOTALDEMAND'], adjusted=False)

# # Plot CCF
# plt.figure(figsize=(10,6))
# plt.bar(np.arange(len(cross_corr)), cross_corr)
# plt.title('Cross-Correlation between Temperature and Electricity Demand')
# plt.xlabel('Lag')
# plt.ylabel('Cross-Correlation')
# plt.show()


# Test stationarity
from statsmodels.tsa.stattools import adfuller

# Perform the ADF test on the 'TOTALDEMAND' column
result = adfuller(cleaned_data['TOTALDEMAND'])

# Print the results
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

if result[1] < 0.05:
    print("The data is stationary (reject the null hypothesis)")
else:
    print("The data is non-stationary (fail to reject the null hypothesis)")
    
# Compute the pacf and select the top 8 most influential lags
from statsmodels.tsa.stattools import pacf
# Step 1: Compute PACF values using Yule-Walker method
# We assume your time series is stored in the 'TOTALDEMAND' column
pacf_values = pacf(cleaned_data['TOTALDEMAND'], nlags=60, method='ywm')

# Step 2: Identify the top 5 influential lags (considering both positive and negative correlations)
# Create a DataFrame to store lag and PACF values
pacf_df = pd.DataFrame({'Lag': np.arange(len(pacf_values)), 'PACF': pacf_values})

# Sort by the absolute value of PACF (to capture both positive and negative influences)
pacf_df['PACF_abs'] = pacf_df['PACF'].abs()

# Select the top 5 lags with the highest absolute PACF values, excluding lag 0
top_8_lags = pacf_df.iloc[1:].nlargest(10, 'PACF_abs')['Lag'].values  # Exclude lag 0

print("Top 5 Influential Lags:", top_8_lags)

# Step 3: Create lagged variables for the top 5 influential lags
for lag in top_5_lags:
    cleaned_data[f'Lag{int(lag)}'] = cleaned_data['TOTALDEMAND'].shift(int(lag))

# Create lagged variables for the top 5 influential lags from PACF
cleaned_data['Lag1'] = cleaned_data['TOTALDEMAND'].shift(1)  # 30 minutes ago
cleaned_data['Lag2'] = cleaned_data['TOTALDEMAND'].shift(2)  # 1 hour ago
cleaned_data['Lag41'] = cleaned_data['TOTALDEMAND'].shift(41)  # 1.5 hours ago
cleaned_data['Lag47'] = cleaned_data['TOTALDEMAND'].shift(47)  # 12 hours ago
cleaned_data['Lag48'] = cleaned_data['TOTALDEMAND'].shift(48)  # 24 hours ago

# Drop NaN values that result from shifting the data
cleaned_data.dropna(inplace=True)

# One-hot encode the 'DayOfWeek', 'Month', and 'DayType' columns
cleaned_data = pd.get_dummies(cleaned_data, columns=['DayOfWeek', 'Month', 'DayType'], drop_first=True)

# Feature selection to eliminate highly correlated and redundant features
import seaborn as sns
import matplotlib.pyplot as plt

# Check correlation between features
corr_matrix = cleaned_data[['TEMPERATURE', 'Hour', 
                            'Lag1', 'Lag2', 'Lag41', 'Lag47', 'Lag48']].corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

############################## Training 1D-CNN
cleaned_data2 = cleaned_data.copy()
cleaned_data2['time_sin'] = np.sin(2 * np.pi * cleaned_data2['Hour'] / 24)
cleaned_data2 = cleaned_data2.sort_values(by='DATETIME')

# Create rolling mean, min, and max features for the past 4 and 8 hours
# Add temperature^2
cleaned_data2['TempSquared'] = cleaned_data2['TEMPERATURE'] ** 2
cleaned_data2['Temp_mean_8h'] = cleaned_data2['TEMPERATURE'].rolling(window=16, min_periods=1).mean()  # 16 intervals for 8 hours (30-min interval data)
cleaned_data2['Temp_min_8h'] = cleaned_data2['TEMPERATURE'].rolling(window=16, min_periods=1).min()
cleaned_data2['Temp_max_8h'] = cleaned_data2['TEMPERATURE'].rolling(window=16, min_periods=1).max()
cleaned_data2['Temp_mean_4h'] = cleaned_data2['TEMPERATURE'].rolling(window=8, min_periods=1).mean()  # 48 intervals for 24 hours
cleaned_data2['Temp_min_4h'] = cleaned_data2['TEMPERATURE'].rolling(window=8, min_periods=1).min()
cleaned_data2['Temp_max_4h'] = cleaned_data2['TEMPERATURE'].rolling(window=8, min_periods=1).max()
cleaned_data2['EWMA_4h'] = cleaned_data2['TOTALDEMAND'].ewm(span=8, adjust=False).mean()
cleaned_data2['EWMA_8h'] = cleaned_data2['TOTALDEMAND'].ewm(span=16, adjust=False).mean()

cleaned_data2 = cleaned_data2.drop(columns=['DATETIME', 'REGIONID', 'LOCATION', 'Date', 'DateOnly', 'Hour', 'Year'])
cleaned_data2 = cleaned_data2.dropna()

# Perform Min-max scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Separate the continuous and categorical columns
continuous_cols = ['TEMPERATURE', 'time_sin', 'TempSquared', 'Lag1', 'Lag2', 'Lag48', 'Lag47', 'Lag41', 
                   'EWMA_4h', 'EWMA_8h', 'Temp_mean_8h', 'Temp_min_8h', 'Temp_max_8h', 
                   'Temp_mean_4h', 'Temp_min_4h', 'Temp_max_4h', 'Humidity', 'SolarRadiation']

# Split the data into features (X) and target (Y)
X = cleaned_data2.drop(columns=['TOTALDEMAND'])  # Features
y = cleaned_data2['TOTALDEMAND']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 3: Initialize and apply separate scalers for X (features) and Y (target)

# X scaler (for independent variables)
scaler_X = MinMaxScaler()
X_train[continuous_cols] = scaler_X.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler_X.transform(X_test[continuous_cols])

# Y scaler (for dependent variable)
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Reshape X for 1D-CNN (samples, timesteps, features)
X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define a basic 1D CNN Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Define the 1D CNN model
def create_1d_cnn(input_shape):
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())  # Normalize for faster learning
    model.add(MaxPooling1D(pool_size=2))  # Down-sample

    # 2nd Convolutional Layer
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Flatten the output before passing to Dense layers
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(1))  # Output layer to predict the next 30-minute demand

    # Compile the model
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

    return model

# Get the input shape (features for 1D-CNN)
input_shape = (X_train.shape[1], X_train.shape[2])

# Create the model
model = create_1d_cnn(input_shape)

# Train the model
base_model = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
model.save('base_model/')

# compute error metrics on test set
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error

y_pred_scaled = model.predict(X_test)
# Inverse transform the predictions back to the original scale
y_pred_original_scale = scaler_y.inverse_transform(y_pred_scaled)
# Inverse transform the test target as well for comparison
y_test_original_scale = scaler_y.inverse_transform(y_test_scaled)
# Now you can evaluate the predictions in the original scale

mae = mean_absolute_error(y_test_original_scale, y_pred_original_scale)
print(f"MAE (original scale): {mae}")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute RMSE, MAPE, MAE, MSLE, and MSE

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Mean Squared Logarithmic Error (MSLE)
msle = mean_squared_log_error(y_test, y_pred)

# Print all the metrics
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print(f'MSLE: {msle}')

import pickle

# Save the training history (loss and accuracy per epoch)
with open('history.pkl', 'wb') as file:
    pickle.dump(history.history, file)


# Manual testing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Define the 1D CNN model with the specified architecture
def create_cnn_model():
    model = Sequential()
    
    # Layer 1: Conv1D with 32 filters, kernel size 2, ReLU activation
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
    
    # Layer 2: Conv1D with 64 filters, kernel size 2, ReLU activation
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    
    # Layer 3: Conv1D with 128 filters, kernel size 2, ReLU activation
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    
    # Layer 4: Conv1D with 256 filters, kernel size 2, ReLU activation
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu'))
    
    # Max Pooling layer with pool size 3
    model.add(MaxPooling1D(pool_size=3))
    
    # Flatten the output to feed it into Dense layers
    model.add(Flatten())
    
    # Dense layer 1 with 400 nodes
    model.add(Dense(400, activation='relu'))
    
    # Dense layer 2 with 400 nodes
    model.add(Dense(400, activation='relu'))
    
    # Dense layer 3 with 200 nodes
    model.add(Dense(200, activation='relu'))
    
    # Output layer with 1 node (for regression)
    model.add(Dense(1))
    
    # Compile the model with Adam optimizer and a learning rate of 0.0001
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model

# Step 2: Create the model
model_test = create_cnn_model()

# Step 3: Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 4: Train the model with early stopping
Test_model = model_test.fit(X_train, y_train_scaled, 
                    epochs=50, 
                    batch_size=128, 
                    validation_split=0.2, 
                    callbacks=[early_stopping], 
                    verbose=1)

# Step 5: Evaluate the model on the test set
test_loss, test_mae = model_test.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

y_pred_scaled = model_test.predict(X_test)
y_pred_original_scale = scaler_y.inverse_transform(y_pred_scaled)

# Inverse transform y_test for comparison
y_test_original_scale = scaler_y.inverse_transform(y_test_scaled)

# Evaluate the model using MAE or other metrics
mae = mean_absolute_error(y_test_original_scale, y_pred_original_scale)
mse = mean_squared_error(y_test_original_scale, y_pred_original_scale)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test_original_scale, y_pred_original_scale)
msle = mean_squared_log_error(y_test_original_scale, y_pred_original_scale)
# Print all the metrics
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print(f'MSLE: {msle}')

import kerastuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define the model-building function for Keras Tuner
def build_model(hp):
    model = Sequential()

    # Tuning the number of filters and kernel size for the first Conv1D layer
    model.add(Conv1D(filters=hp.Int('filters_1', min_value=32, max_value=128, step=32),
                     kernel_size=hp.Choice('kernel_size_1', values=[2, 3, 5]),
                     activation='relu', input_shape=(X_train.shape[1], 1)))

    # Second Conv1D layer with tunable filters and kernel size
    model.add(Conv1D(filters=hp.Int('filters_2', min_value=32, max_value=128, step=32),
                     kernel_size=hp.Choice('kernel_size_2', values=[2, 3, 5]),
                     activation='relu'))

    # Third Conv1D layer
    model.add(Conv1D(filters=hp.Int('filters_3', min_value=64, max_value=256, step=64),
                     kernel_size=hp.Choice('kernel_size_3', values=[2, 3, 5]),
                     activation='relu'))

    # Fourth Conv1D layer
    model.add(Conv1D(filters=hp.Int('filters_4', min_value=64, max_value=256, step=64),
                     kernel_size=hp.Choice('kernel_size_4', values=[2, 3, 5]),
                     activation='relu'))

    # Max Pooling layer with tunable pool size
    model.add(MaxPooling1D(pool_size=hp.Choice('pool_size', values=[2, 3])))

    # Flatten the output to feed into dense layers
    model.add(Flatten())

    # Tuning the number of units in the first dense layer
    model.add(Dense(units=hp.Int('dense_units_1', min_value=128, max_value=512, step=128), activation='relu'))

    # Second dense layer with tunable units
    model.add(Dense(units=hp.Int('dense_units_2', min_value=128, max_value=512, step=128), activation='relu'))

    # Third dense layer with 200 units as per your specification
    model.add(Dense(200, activation='relu'))

    # Dropout layer with tunable dropout rate
    model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))

    # Output layer for regression
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
                  loss='mean_squared_error', metrics=['mae'])

    return model

# Step 2: Set up the Keras Tuner RandomSearch
tuner = kt.RandomSearch(
    build_model,
    objective = 'val_mae',  # Optimizing for validation MAE
    max_trials = 5,  # Number of hyperparameter sets to try
    executions_per_trial = 1,  # Number of models to train with each set of hyperparameters
    directory='tuner_dir',
    project_name='1d_cnn_tuning')

# Step 3: Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=50, batch_size=128, validation_split=0.2, verbose=1)

# Step 4: Retrieve the best model and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Step 5: Evaluate the best model on the test set
test_loss, test_mae = best_model.evaluate(X_test, y_test)
print(f"Best Test MAE: {test_mae}")
print(f"Best Hyperparameters: {best_hyperparameters.values}")

# Explore LSTM with similar model complexity and variables

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define the LSTM model architecture
def create_lstm_model(input_shape):
    model = Sequential()

    # First LSTM layer with 64 units and ReLU activation
    model.add(LSTM(units=64, activation='relu', return_sequences=True, input_shape=input_shape))

    # Second LSTM layer with 128 units
    model.add(LSTM(units=128, activation='relu', return_sequences=False))

    # Dropout for regularization
    model.add(Dropout(0.3))

    # Dense layer with 100 nodes
    model.add(Dense(100, activation='relu'))

    # Output layer with linear activation for regression
    model.add(Dense(1, activation='linear'))

    # Compile the model with Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])
    
    return model

# Create the model
sequence_length = 48

# Create and train the LSTM model
input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
lstm_model = create_lstm_model(input_shape)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = lstm_model.fit(X_train, y_train_scaled, epochs=50, batch_size=128, validation_split=0.2, verbose=1,
                         callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_mae = lstm_model.evaluate(X_test, y_test_scaled)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Predict on the test set
y_pred_scaled = lstm_model.predict(X_test)

# Inverse transform the predictions to original scale
y_pred_original_scale = scaler_y.inverse_transform(y_pred_scaled)

# Inverse transform y_test for comparison
y_test_original_scale = scaler_y.inverse_transform(y_test_scaled)

# Calculate MAE in the original scale
from sklearn.metrics import mean_absolute_error
mae_original = mean_absolute_error(y_test_original_scale, y_pred_original_scale)
print(f"MAE (original scale): {mae_original}")




