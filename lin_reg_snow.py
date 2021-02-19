import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Linear regression that predicts the next day's amount of snow using temperature, precipitation and previous days snow
# This code produces to models. One uses the previous days snow depth and the other doesn't.

# ------------------- Instructions ------------------- #
# 1. Change 'station_name' below to your station (line 15)
# 2. Change the file location (line 56)
# ---------------------------------------------------- #

station_name = 'Fielding Lake (1268)'
column_name = station_name + ' Snow Depth (cm) Start of Day Values'

def create_df_shift(file_name, column_name):  # Creating a dataframe using a csv
    df = pd.read_csv(file_name)  # df stands for dataframe
    shifted = df[column_name]  # Creating an extra column containing the previous days snow depth
    shifted = shifted.shift(periods=1)
    df['Prev Snow Depth (cm)'] = shifted  # Adds previous snow levels to dataframe
    df = df.dropna()  # Removes null values
    return df


def create_df(file_name):  # Creating a dataframe using a csv
    df = pd.read_csv(file_name)  # df stands for dataframe
    df = df.dropna()  # Removes null values
    return df


def test_train_split(df, column_name):
    y = df[column_name]  # Assign snow depth to y-axis (labels)
    X = df.drop([column_name, 'Date'], axis=1)  # Removes date and snow depth column
    train_size = int(round(len(X) * 0.6, 0))  # Training data is 60% of data
    X_train, X_test, y_train, y_test = X.iloc[0:train_size], X.iloc[train_size+1:-1], y.iloc[0:train_size], y.iloc[train_size+1:-1]
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    return X_train, X_test, y_train, y_test


def plot_nn(predictions_s, y_test, predictions_n):
    X = np.linspace(0, 1, len(y_test))  # NEED TO CHANGE THIS FOR DATES
    plt.figure(figsize=(8, 5))
    plt.plot(X, predictions_s, 'r-', label='Shifted Linear regression')
    plt.plot(X, predictions_n, 'g-', label='Non Shifted Linear regression')
    plt.plot(X, y_test, 'b-', label='Measured')
    plt.legend()
    plt.ylabel('Snow Depth (cm)')
    plt.xlabel('Date')
    plt.title('Fielding Lake Snow Depth')
    plt.show()


def lin_reg_plot(column_name, plot):
    file_name = 'D:/Users/Joshg/Documents/MDM3/Alaska/Fielding_Lake_1268_clean.csv'  # File location
    df = create_df_shift(file_name, column_name)

    X_train_s, X_test_s, y_train, y_test = test_train_split(df, column_name)  # _s at the end for shifted
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
        plot_nn(prediction_s, y_test, prediction_n)

    return rms_s, rms_n, reg_score_s, reg_score_n


rms_s, rms_n, reg_score_s, reg_score_n = lin_reg_plot(column_name, 'yes')
print('--------- Using previous snow depth -------------')
print('rms is ' + str(rms_s))  # Calculating and printing mean rms
print('reg score is ' + str(reg_score_s))  # This is the score calculated by sklearn

print('------- Not using previous snow depth ------------')
print('rms is ' + str(rms_n))  # Calculating and printing mean rms
print('reg score is ' + str(reg_score_n))  # This is the score calculated by sklearn
