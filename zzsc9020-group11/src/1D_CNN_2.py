# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:24:43 2024

@author: Jack
"""
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
pub_holiday['Date'] = pd.to_datetime(pub_holiday['Date'], format = '%Y/%m/%d').dt.date

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

# Split the data into features (X) and target (Y)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X = cleaned_data[['TEMPERATURE', 'IsHoliday',
       'Lag1', 'Lag2', 'Lag41', 'Lag47', 'Lag48', 'hour_sin', 'hour_cos',
       'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos',
       'dayofyear_sin', 'dayofyear_cos', 'TempSquared', 'Temp_mean_8h',
       'Temp_min_8h', 'Temp_max_8h', 'Temp_mean_4h', 'Temp_min_4h',
       'Temp_max_4h', 'EWMA_4h', 'EWMA_8h']]  # Features
y = cleaned_data['TOTALDEMAND']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 2: Combine X and y for scaling
combined_train = pd.concat([X_train, y_train], axis=1)  # Combine features and target for training
combined_test = pd.concat([X_test, y_test], axis=1)  # Combine features and target for testing

# Step 3: Apply MinMaxScaler to the combined training data
scaler = MinMaxScaler()

# Fit the scaler on the combined training data
combined_train_scaled = scaler.fit_transform(combined_train)

# Transform the test data using the same scaler
combined_test_scaled = scaler.transform(combined_test)

# Step 4: Separate X and y after scaling
X_train_scaled = combined_train_scaled[:, :-1]  # Scaled features for training (TEMPERATURE, TOTALDEMAND as feature)
y_train_scaled = combined_train_scaled[:, -1:]  # Scaled target for training (TOTALDEMAND)

X_test_scaled = combined_test_scaled[:, :-1]  # Scaled features for testing
y_test_scaled = combined_test_scaled[:, -1:]  # Scaled target for testing

# Build 1D CNN
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
    # model.add(Dropout(0.5))
    
    # Dense layer 2 with 400 nodes
    model.add(Dense(400, activation='relu'))
    # model.add(Dropout(0.2))
    
    # Dense layer 3 with 200 nodes
    model.add(Dense(200, activation='relu'))
    # Dense layer 3 with 100 nodes
    model.add(Dense(100, activation='relu'))
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
Test_model = model_test.fit(X_train_scaled, y_train_scaled, 
                    epochs=50, 
                    batch_size=128, 
                    validation_split=0.2, 
                    callbacks=[early_stopping], 
                    verbose=1)

# Step 5: Evaluate the model on the test set
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error

# Predict using the model
y_pred_scaled = model_test.predict(X_test_scaled)

# Inverse transform the scaled predictions and actual y_test using the same scaler
# We need to concatenate zeros to simulate the feature columns that were scaled together with the target
y_test_rescaled = scaler.inverse_transform(np.concatenate([X_test_scaled, y_test_scaled], axis=1))[:, -1]
y_pred_rescaled = scaler.inverse_transform(np.concatenate([X_test_scaled, y_pred_scaled], axis=1))[:, -1]


# Evaluate the model using MAE or other metrics
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)
msle = mean_squared_log_error(y_test_rescaled, y_pred_rescaled)

# Print all the metrics
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print(f'MSLE: {msle}')


model.save('Test_model/')
