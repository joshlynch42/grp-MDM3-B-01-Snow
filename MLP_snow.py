from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import math

# Neural network that predicts the next day's amount of snow using temperature, precipitation and previous days snow
# This model uses the previous day's amount of snow to calculate the following day's. Comment out lines 25-27 to get a
# result without the previous days data

# ------------------- Instructions ------------------- #
# 1. Change 'station_name' below to your station (line 21)
# 2. Change the file location (line 73)
# 3. Change the number of iteration you would like the code to take when calculating the mean rms. Don't go too high as
#    the code is very slow. (try between 1-10) (line 100)
# ---------------------------------------------------- #

station_name = 'Alexander Lake (1267)'
column_name = station_name + ' Snow Depth (cm) Start of Day Values'


def create_df(file_name, column_name):  # Creating a dataframe using a csv
    df = pd.read_csv(file_name)  # df stands for dataframe
    shifted = df[column_name]  # Creating an extra column containing the previous days snow depth
    shifted = shifted.shift(periods=-90)
    df['Prev Snow Depth (cm)'] = shifted  # Adds previous snow levels to dataframe
    df = df.dropna()  # Removes null values
    return df


def test_train_split(df, column_name, station_name):
    y = df[column_name]  # Assign snow depth to y-axis (labels)
    X = df.drop([column_name, 'Date'], axis=1)  # Removes date and snow depth columns
    # X = tanh_df(X, station_name)
    train_size = int(round(len(X) * 0.6, 0))  # Training data is 60% of data
    X_train, X_test, y_train, y_test = X.iloc[0:train_size], X.iloc[train_size + 1:-1], y.iloc[0:train_size], y.iloc[
                                                                                                  train_size + 1:-1]
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    min_date = df['Date'].iloc[train_size]
    date_time_obj = datetime.datetime.strptime(min_date, '%Y-%m-%d')
    max_date = date_time_obj + timedelta(days=len(y_test)-1)
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


def scaling_values(X_train, X_test):  # Scaling the train values
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def plot_nn(predictions_s, y_test, predictions_n, time_axis, station_name):
    plt.figure(figsize=(10, 7))
    plt.plot(time_axis, predictions_s, 'g-', label='Predicting Snow 3 Months In Advance')
    plt.plot(time_axis, predictions_n, 'r-', label='Predicting Snow Using Temperature and Precipitation')
    plt.plot(time_axis, y_test, 'b-', label='Measured Data')
    plt.legend(loc='best')
    plt.ylabel('Snow Depth (cm)')
    plt.xlabel('Date')
    plt.gcf().autofmt_xdate()
    plt.title(station_name + ' Snow Depth using a MLP Neural Network')
    plt.show()


def MLP_fit(column_name, plot, station_name):
    file_name = 'D:/Users/Joshg/Documents/MDM3/Alaska/Alexander_Lake_1267_clean.csv'  # File location
    df = create_df(file_name, column_name)

    X_train_s, X_test_s, y_train, y_test, time_axis = test_train_split(df, column_name, station_name)  # _s at the end for shifted
    X_train_n, X_test_n = np.delete(X_train_s, 6, 1), np.delete(X_test_s, 6, 1)  # _n at the end for not shifted

    X_train_s, X_test_s = scaling_values(X_train_s, X_test_s)
    X_train_n, X_test_n = scaling_values(X_train_n, X_test_n)

    MLP = MLPRegressor(hidden_layer_sizes=(20, 8, 6, 5), max_iter=2000)  # Creating an MLP neural network
    MLP.fit(X_train_s, y_train)  # Fitting neural network using train values
    predictions_s = MLP.predict(X_test_s)  # Predicting values for the test data
    rms_s = mean_squared_error(y_test, predictions_s, squared=False)  # Calculating root mean squared error

    MLP = MLPRegressor(hidden_layer_sizes=(20, 8, 6, 5), max_iter=2000)  # Creating an MLP neural network
    MLP.fit(X_train_n, y_train)  # Fitting neural network using train values
    predictions_n = MLP.predict(X_test_n)  # Predicting values for the test data
    rms_n = mean_squared_error(y_test, predictions_n, squared=False)  # Calculating root mean squared error

    if plot == 'yes':
        plot_nn(predictions_s, y_test, predictions_n, time_axis, station_name)
    return rms_s, rms_n


# Using a for loop to calculate average rms
total_rms_s = 0  # Initialise a total counter
total_rms_n = 0
num_it = 5  # Number of iterations
for i in range(num_it):
    print('Iteration number ' + str(i))
    if i < num_it - 1:
        rms_s, rms_n = MLP_fit(column_name, 'no', station_name)  # Adding rms to mean total
        total_rms_s += rms_s
        total_rms_n += rms_n
    else:
        rms_s, rms_n = MLP_fit(column_name, 'yes', station_name)  # Plotting the last iteration of NN as an example
        total_rms_s += rms_s
        total_rms_n += rms_n

print('--------- Predicting Snow 3 Months In Advance -------------')
print('average rms is ' + str(total_rms_s/num_it))  # Calculating and printing mean rms
print('----- Predicting Snow Using Temperature and Precipitation -----')
print('average rms is ' + str(total_rms_n/num_it))  # Calculating and printing mean rms
