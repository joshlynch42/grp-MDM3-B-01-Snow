import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import datetime
from datetime import timedelta
import math

# Linear regression that predicts the next day's amount of snow using temperature, precipitation and previous days snow
# This code produces to models. One uses the previous days snow depth and the other doesn't.

# ------------------- Instructions ------------------- #
# 1. Change 'station_name' below to your station (line 17)
# 2. Change the file location (line 63)
# ---------------------------------------------------- #

station_name = 'Alexander Lake (1267)'
column_name = station_name + ' Snow Depth (cm) Start of Day Values'

def create_df_shift(file_name, column_name):  # Creating a dataframe using a csv
    df = pd.read_csv(file_name)  # df stands for dataframe
    shifted = df[column_name]  # Creating an extra column containing the previous days snow depth
    shifted = shifted.shift(periods=90)
    df['Prev Snow Depth (cm)'] = shifted  # Adds previous snow levels to dataframe
    df = df.dropna()  # Removes null values
    return df


def create_df(file_name):  # Creating a dataframe using a csv
    df = pd.read_csv(file_name)  # df stands for dataframe
    df = df.dropna()  # Removes null values
    return df


def test_train_split(df, column_name, station_name):
    y = df[column_name]  # Assign snow depth to y-axis (labels)
    X = df.drop([column_name, 'Date'], axis=1)  # Removes date and snow depth column
    # X = tanh_df(X, station_name)
    train_size = int(round(len(X) * 0.6, 0))  # Training data is 60% of data
    X_train, X_test, y_train, y_test = X.iloc[0:train_size], X.iloc[train_size+1:-1], y.iloc[0:train_size], y.iloc[train_size+1:-1]
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    min_date = df['Date'].iloc[train_size]
    date_time_obj = datetime.datetime.strptime(min_date, '%Y-%m-%d')
    max_date = date_time_obj + timedelta(days=len(y_test) - 1)
    time_axis = pd.date_range(start=min_date, end=max_date)
    return X_train, X_test, y_train, y_test, time_axis


def tanh_df(df, station_name):
    tanh = []
    scaled = (df[station_name + ' Air Temperature Average (degC)']/df[station_name + ' Air Temperature Average (degC)'].max()) * 5
    scaled = pd.DataFrame(scaled)
    for index, row in scaled.iterrows():
        value = math.tanh(row[station_name + ' Air Temperature Average (degC)'])
        tanh.append(value)
    df['tanh'] = tanh
    return df


def plot_nn(predictions_s, y_test, predictions_n, time_axis, station_name):
    plt.figure(figsize=(10, 7))
    plt.plot(time_axis, predictions_s, 'g-', label='Predicting Snow 3 Months In Advance')
    plt.plot(time_axis, predictions_n, 'r-', label='Predicting Snow Using Temperature and Precipitation')
    plt.plot(time_axis, y_test, 'b-', label='Measured Data')
    plt.legend(loc=2)
    plt.ylabel('Snow Depth (cm)')
    plt.xlabel('Date')
    plt.gcf().autofmt_xdate()
    plt.title(station_name + ' Snow Depth using Linear Regression')
    plt.show()


def lin_reg_plot(column_name, plot, station_name):
    file_name = 'D:/Users/Joshg/Documents/MDM3/Alaska/Alexander_Lake_1267_clean.csv'  # File location
    df = create_df_shift(file_name, column_name)

    X_train_s, X_test_s, y_train, y_test, time_axis = test_train_split(df, column_name, station_name)  # _s at the end for shifted
    X_train_n, X_test_n = np.delete(X_train_s, 6, 1), np.delete(X_test_s, 6, 1)  # _n at the end for not shifted

    reg_s = LinearRegression().fit(X_train_s, y_train)  # Creating a linear regression
    prediction_s = reg_s.predict(X_test_s)  # Creating predictions using test data

    reg_n = LinearRegression().fit(X_train_n, y_train)  # Creating a linear regression
    prediction_n = reg_n.predict(X_test_n)  # Creating predictions using test data

    rms_s = mean_squared_error(y_test, prediction_s, squared=False)
    reg_score_s = reg_s.score(X_train_s, y_train)

    rms_n = mean_squared_error(y_test, prediction_n, squared=False)
    reg_score_n = reg_n.score(X_train_n, y_train)

    if plot == 'yes':
        plot_nn(prediction_s, y_test, prediction_n, time_axis, station_name)

    return rms_s, rms_n, reg_score_s, reg_score_n


rms_s, rms_n, reg_score_s, reg_score_n = lin_reg_plot(column_name, 'yes', station_name)
print('--------- Predicting Snow 3 Months In Advance -------------')
print('rms is ' + str(rms_s))  # Calculating and printing mean rms
print('reg score is ' + str(reg_score_s))  # This is the score calculated by sklearn

print('----- Predicting Snow Using Temperature and Precipitation -----')
print('rms is ' + str(rms_n))  # Calculating and printing mean rms
print('reg score is ' + str(reg_score_n))  # This is the score calculated by sklearn
