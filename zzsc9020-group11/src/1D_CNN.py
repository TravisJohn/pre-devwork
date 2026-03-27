# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:24:43 2024

@author: Jack
"""

import os
import pandas as pd
import numpy as np
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
# pub_holiday['Date'] = pd.to_datetime(pub_holiday['Date'], format = '%Y/%m/%d').dt.date
pub_holiday['Date'] = pd.to_datetime(pub_holiday['Date'], format='%Y-%m-%d').dt.date

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

# Fill NA values in 'Humidity' and 'SolarRadiation' columns by propagating the previous rows' values (forward fill)
cleaned_data['Humidity'] = cleaned_data['Humidity'].interpolate(method='linear', limit=4)
cleaned_data['SolarRadiation'] = cleaned_data['SolarRadiation'].interpolate(method='linear', limit=4)

# Update 'DayType' to include Public Holiday
cleaned_data['Lag1'] = cleaned_data['TOTALDEMAND'].shift(1)  # 30 minutes ago
cleaned_data['Lag2'] = cleaned_data['TOTALDEMAND'].shift(2)  # 1 hour ago
cleaned_data['Lag41'] = cleaned_data['TOTALDEMAND'].shift(41)  # 1.5 hours ago
cleaned_data['Lag47'] = cleaned_data['TOTALDEMAND'].shift(47)  # 12 hours ago
cleaned_data['Lag48'] = cleaned_data['TOTALDEMAND'].shift(48)  # 24 hours ago

# Drop NaN values that result from shifting the data
cleaned_data.dropna(inplace=True)

# Cyclical encoding
cleaned_data['hour_min'] = cleaned_data['DATETIME'].dt.hour + cleaned_data['DATETIME'].dt.minute / 60
cleaned_data['DayOfYear'] = cleaned_data['DATETIME'].dt.dayofyear

# Cyclical encoding for hour_min (hours of the day)
cleaned_data['hour_sin'] = np.sin(2 * np.pi * cleaned_data['hour_min'] / 24)
cleaned_data['hour_cos'] = np.cos(2 * np.pi * cleaned_data['hour_min'] / 24)

# Cyclical encoding for DayOfWeek (days of the week)
cleaned_data['dayofweek_sin'] = np.sin(2 * np.pi * cleaned_data['DayOfWeek'] / 7)
cleaned_data['dayofweek_cos'] = np.cos(2 * np.pi * cleaned_data['DayOfWeek'] / 7)

# Cyclical encoding for Month (months of the year)
cleaned_data['month_sin'] = np.sin(2 * np.pi * cleaned_data['Month'] / 12)
cleaned_data['month_cos'] = np.cos(2 * np.pi * cleaned_data['Month'] / 12)

# Cyclical encoding for DayOfYear (days of the year)
cleaned_data['dayofyear_sin'] = np.sin(2 * np.pi * cleaned_data['DayOfYear'] / 365)
cleaned_data['dayofyear_cos'] = np.cos(2 * np.pi * cleaned_data['DayOfYear'] / 365)

# One-hot encode the 'DayOfWeek', 'Month' columns
# cleaned_data = pd.get_dummies(cleaned_data, columns=['DayOfWeek', 'Month'], drop_first=True)

cleaned_data['TempSquared'] = cleaned_data['TEMPERATURE'] ** 2
cleaned_data['Temp_mean_8h'] = cleaned_data['TEMPERATURE'].rolling(window=16, min_periods=1).mean()  # 16 intervals for 8 hours (30-min interval data)
cleaned_data['Temp_min_8h'] = cleaned_data['TEMPERATURE'].rolling(window=16, min_periods=1).min()
cleaned_data['Temp_max_8h'] = cleaned_data['TEMPERATURE'].rolling(window=16, min_periods=1).max()
cleaned_data['Temp_mean_4h'] = cleaned_data['TEMPERATURE'].rolling(window=8, min_periods=1).mean()  # 48 intervals for 24 hours
cleaned_data['Temp_min_4h'] = cleaned_data['TEMPERATURE'].rolling(window=8, min_periods=1).min()
cleaned_data['Temp_max_4h'] = cleaned_data['TEMPERATURE'].rolling(window=8, min_periods=1).max()
cleaned_data['EWMA_4h'] = cleaned_data['TOTALDEMAND'].ewm(span=8, adjust=False).mean()
cleaned_data['EWMA_8h'] = cleaned_data['TOTALDEMAND'].ewm(span=16, adjust=False).mean()

cleaned_data = cleaned_data.drop(columns=['DATETIME', 'REGIONID', 'LOCATION', 'Date', 'DateOnly', 'Hour', 'Year', 
                                           'hour_min', 'DayOfWeek', 'Month', 'DayOfYear'])


continuous_cols = ['TEMPERATURE', 'TempSquared', 'Lag1', 'Lag2', 'Lag48', 'Lag47', 'Lag41']


# Split the data into features (X) and target (Y)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X = cleaned_data[['TEMPERATURE', 'IsHoliday',
       'Lag1', 'Lag2', 'Lag41', 'Lag47', 'Lag48', 'hour_sin', 'hour_cos',
       'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos',
       'dayofyear_sin', 'dayofyear_cos', 'TempSquared'
       ]]  # Features
y = cleaned_data['TOTALDEMAND']  # Target variable

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

# Build 1D CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
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
    # model.add(Dropout(0.5))
    
    # Dense layer 2 with 400 nodes
    model.add(Dense(400, activation='relu'))
    # model.add(Dropout(0.2))
    
    # Dense layer 3 with 200 nodes
    model.add(Dense(200, activation='relu'))
    # Dense layer 3 with 100 nodes
    # model.add(Dense(100, activation='relu'))
    # Dense layer 4 with 50 nodes
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
                    epochs=500, 
                    batch_size=128, 
                    validation_split=0.2, 
                    callbacks=[early_stopping], 
                    verbose=1)

# Step 5: Evaluate the model on the test set
test_loss, test_mae = model_test.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error
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

# Assuming 'Test_model' is your trained model and contains the history of training
history = Test_model.history
import matplotlib.pyplot as plt
# Plot the training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history['loss'], label='Training Loss', color='blue')
plt.plot(history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

model_test.save('Test_model/')

from tensorflow.keras.models import load_model

# Load the saved model
model_test = load_model('Test_model/')


# Load and test the model
from tensorflow.keras.models import load_model
CNN_model = load_model('Test_model/')
X_test_padded = np.pad(X_test, ((0, 0), (0, 12), (0, 0)), 'constant')
y_pred = CNN_model.predict(X_test_padded)

# Evaluate the model on test data
test_loss, test_mae = model_test.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error
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
