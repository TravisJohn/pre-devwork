# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 00:50:06 2024

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

# Fill NA values in 'Humidity' and 'SolarRadiation' columns by propagating the previous rows' values (forward fill)
cleaned_data['Humidity'] = cleaned_data['Humidity'].interpolate(method='linear', limit=4)
cleaned_data['SolarRadiation'] = cleaned_data['SolarRadiation'].interpolate(method='linear', limit=4)

# Update 'DayType' to include Public Holiday
cleaned_data['Lag1'] = cleaned_data['TOTALDEMAND'].shift(1)  # 30 minutes ago
cleaned_data['Lag2'] = cleaned_data['TOTALDEMAND'].shift(2)  # 1 hour ago
cleaned_data['Lag34'] = cleaned_data['TOTALDEMAND'].shift(34)
cleaned_data['Lag36'] = cleaned_data['TOTALDEMAND'].shift(36)
cleaned_data['Lag39'] = cleaned_data['TOTALDEMAND'].shift(39)
cleaned_data['Lag41'] = cleaned_data['TOTALDEMAND'].shift(41)  # 1.5 hours ago
cleaned_data['Lag47'] = cleaned_data['TOTALDEMAND'].shift(47)  # 12 hours ago
cleaned_data['Lag48'] = cleaned_data['TOTALDEMAND'].shift(48)  # 24 hours ago
cleaned_data['Lag49'] = cleaned_data['TOTALDEMAND'].shift(49)
cleaned_data['Lag50'] = cleaned_data['TOTALDEMAND'].shift(50)

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

X = cleaned_data[['TEMPERATURE',
       'Lag1', 'Lag2', 'Lag39','Lag41', 'Lag47', 'Lag48', 'Lag49', 'Lag50', 
       'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos',
       'dayofyear_sin', 'dayofyear_cos', 'TempSquared'
       ]]  # Features

y = cleaned_data['TOTALDEMAND']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train XGBoost Model
import xgboost as xgb

# Using best params
xgb_model = xgb.XGBRegressor(
    n_estimators=900,
    learning_rate=0.12,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=1,
    min_child_weight=3,
    random_state=42,
    tree_method='hist',
    device ='cuda'  # Enables GPU training
)

# Train the model with early stopping
xgb_model.fit(X_train, y_train.ravel(), verbose=False)
y_pred_xgb = xgb_model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error
mae = mean_absolute_error(y_test, y_pred_xgb)
mse = mean_squared_error(y_test, y_pred_xgb)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred_xgb)
msle = mean_squared_log_error(y_test, y_pred_xgb)
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print(f'MSLE: {msle}')


from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 500, 1000],  # Number of boosting rounds
    'learning_rate': [0.01, 0.05, 0.1],  # Learning rate
    'max_depth': [3, 5, 6, 10],  # Maximum depth of a tree
    'subsample': [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
    'colsample_bytree': [0.6, 0.8, 1.0],  # Subsample ratio of columns
    'gamma': [0, 0.1, 0.3],  # Minimum loss reduction to make a split
    'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight (hessian) needed in a child
}

# Initialize the XGBoost regressor
xgb_model = xgb.XGBRegressor(tree_method='hist', device='cuda', random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter settings to try
    scoring='neg_mean_absolute_error',  # Scoring metric (can be changed based on your need)
    cv=3,  # 3-fold cross-validation
    verbose=1,
    random_state=42,
    n_jobs=-1  # Use all cores for parallelization
)

# Fit the RandomizedSearchCV model
random_search.fit(X_train, y_train.ravel())

# Get the best parameters and best score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score (neg MAE): {random_search.best_score_}")

# Train the final model with the best parameters
best_xgb_model = random_search.best_estimator_
best_xgb_model.fit(X_train, y_train.ravel())
y_pred_xgb = best_xgb_model.predict(X_test)


# Second iterations of the fit
param_grid = {
    'n_estimators': [800, 900, 1000, 1100, 1200],  # Around 1000
    'max_depth': [4, 5, 6, 7],  # Around 6
    'learning_rate': [0.08, 0.1, 0.12],  # Around 0.1
    'min_child_weight': [2, 3, 4],  # Around 3
    'gamma': [0, 0.1, 0.2],  # Explore small variations from 0
    'subsample': [0.7, 0.8, 0.9],  # Around 0.8
    'colsample_bytree': [0.9, 1.0],  # Around 1.0
}

from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# Define the model
xgb_model = xgb.XGBRegressor(random_state=42, tree_method='hist', device='cuda')

# RandomizedSearchCV setup
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=20,  # Number of different combinations to try
    scoring='neg_mean_absolute_error',  # Optimize for MAE
    cv=5,  # Cross-validation splits
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Run the random search
random_search.fit(X_train, y_train)

# Print the best parameters found
print(f"Best Parameters: {random_search.best_params_}")

# Get the best estimator from the search
best_xgb_model = random_search.best_estimator_

# Train the model
best_xgb_model.fit(X_train, y_train.ravel())

# Make predictions
y_pred = best_xgb_model.predict(X_test)

# Evaluate the model (using MAE or another metric)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print(f"Final MAE: {mae}")

# 56.19

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

if isinstance(X_test, pd.DataFrame):
    X_test = X_test.values

# Define the number of steps to forecast
n_steps = 2

# Prepare the test data
X_test_original = X_test.copy()
y_test_predicted = []

# Recursive forecasting loop
for step in range(n_steps):
    # Predict one step ahead
    y_pred_step = best_xgb_model.predict(X_test)
    
    # Append the predictions to the list
    y_test_predicted.append(y_pred_step)
    X_test[:, 0] = y_pred_step 
    
# Convert list of predictions into an array
y_test_predicted = np.array(y_test_predicted).flatten()

# Calculate the evaluation metrics for the predicted test set
mae = mean_absolute_error(y_test[:n_steps], y_test_predicted[:n_steps])
rmse = np.sqrt(mean_squared_error(y_test[:n_steps], y_test_predicted[:n_steps]))

# Print the metrics
print(f'Multistep Forecasting - Steps: {n_steps}')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

