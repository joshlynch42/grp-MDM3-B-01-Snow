import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import datetime
from datetime import timedelta

# Linear regression that predicts the next day's amount of snow using temperature, precipitation and previous days snow
# This code produces to models. One uses the previous days snow depth and the other doesn't.

# ------------------- Instructions ------------------- #
# 1. Change 'station_name' below to your station (line 17)
# 2. Change the file location (line 63)
# ---------------------------------------------------- #

station_name = 'Fielding Lake (1268)'
column_name = station_name + ' Snow Depth (cm) Start of Day Values'


def create_df(file_name):  # Creating a dataframe using a csv
    df = pd.read_csv(file_name)  # df stands for dataframe
    df = df.dropna()  # Removes null values
    return df


def temp(y, train_size):
    night_day_temp_diff = 10
    prec_mean = 90
    delta_prec = -5 / 9
    sp = 0
    N = len(y)-1
    t = np.linspace(1, 365*2.5, N+1)
    temp_mean = 5
    delta_temp = 15
    tau = 365
    st = 0

    temp = temp_mean + delta_temp * (np.sin(2 * np.pi * (t - st) / tau)) + (
                (((-1) ** (np.arange(N + 1) % 2 == 0)) * night_day_temp_diff) / 2) + np.random.normal(0, 5, N + 1)

    prec = prec_mean * (1 + delta_prec * (np.sin(2 * np.pi * (t - sp) / tau))) + np.random.normal(0, 5, N + 1)


    temp_train = temp[0: train_size]
    temp_test = temp[train_size+1: -1]
    prec_train = prec[0: train_size]
    prec_test = prec[train_size+1: -1]

    return temp_train, temp_test, prec_train, prec_test



def test_train_split(df, column_name):
    y = df[column_name]  # Assign snow depth to y-axis (labels)
    X = df.drop([column_name, 'Date'], axis=1)  # Removes date and snow depth column
    train_size = int(round(len(X) * 0.8, 0))  # Training data is 60% of data
    y_train, y_test = y.iloc[0:train_size], y.iloc[train_size+1:-1]  # Split snow depth into test and train
    y_train, y_test = np.array(y_train), np.array(y_test)  # Turn test/train into arrays
    temp_train, temp_test, prec_train, prec_test = temp(y, train_size)  # Use maths model to create temp/prec datasets

    X_train = np.vstack([temp_train, prec_train]).T  # Concatenate temp and prec to form test/train data
    X_test = np.vstack([temp_test, prec_test]).T  # Concatenate temp and prec to form test/train data

    min_date = df['Date'].iloc[train_size]
    date_time_obj = datetime.datetime.strptime(min_date, '%Y-%m-%d')
    max_date = date_time_obj + timedelta(days=len(y_test) - 1)
    time_axis = pd.date_range(start=min_date, end=max_date)

    plt.figure(figsize=(8, 5))
    plt.plot(time_axis, temp_test, 'b-', label='Temp')
    plt.legend()
    plt.ylabel('Snow Depth (cm)')
    plt.xlabel('Date')
    plt.gcf().autofmt_xdate()
    plt.title('Fielding Lake Snow Depth')
    plt.show()
    return X_train, X_test, y_train, y_test, time_axis


def plot_nn(predictions, y_test, time_axis):
    plt.figure(figsize=(8, 5))
    plt.plot(time_axis, predictions, 'r-', label='Linear regression with maths')
    plt.plot(time_axis, y_test, 'b-', label='Measured')
    plt.legend()
    plt.ylabel('Snow Depth (cm)')
    plt.xlabel('Date')
    plt.gcf().autofmt_xdate()
    plt.title('Fielding Lake Snow Depth')
    plt.show()


def lin_reg_plot(column_name, plot):
    file_name = 'D:/Users/Joshg/Documents/MDM3/Alaska/Fielding_Lake_1268_clean.csv'  # File location
    df = create_df(file_name)

    X_train, X_test, y_train, y_test, time_axis = test_train_split(df, column_name)  # _s at the end for shifted

    reg = LinearRegression().fit(X_train, y_train)  # Creating a linear regression
    prediction = reg.predict(X_test)  # Creating predictions using test data

    rms = mean_squared_error(y_test, prediction, squared=False)
    reg_score = reg.score(X_train, y_train)

    if plot == 'yes':
        plot_nn(prediction, y_test, time_axis)

    return rms, reg_score


rms, reg_score = lin_reg_plot(column_name, 'yes')
print('--------- Using Maths model -------------')
print('rms is ' + str(rms))  # Calculating and printing mean rms
print('reg score is ' + str(reg_score))  # This is the score calculated by sklearn

