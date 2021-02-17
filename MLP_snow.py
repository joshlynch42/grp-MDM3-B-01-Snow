from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Neural network that predicts the next day's amount of snow using temperature, precipitation and previous days snow
# This model uses the previous day's amount of snow to calculate the following day's. Comment out lines 25-27 to get a
# result without the previous days data

# ------------------- Instructions ------------------- #
# 1. Change 'station_name' below to your station (line 19)
# 2. Change the file location (line 64)
# 3. Change the number of iteration you would like the code to take when calculating the mean rms. Don't go too high as
#    the code is very slow. (try between 1-20) (line 82)
# ---------------------------------------------------- #

station_name = 'Fielding Lake (1268)'
column_name = station_name + ' Snow Depth (cm) Start of Day Values'


def create_df(file_name, column_name):  # Creating a dataframe using a csv
    df = pd.read_csv(file_name)  # df stands for dataframe
    shifted = df[column_name]  # Creating an extra column containing the previous days snow depth
    shifted = shifted.shift(periods=1)
    df['Prev Snow Depth (cm)'] = shifted  # Adds previous snow levels to dataframe
    df = df.dropna()  # Removes null values
    return df


def test_train_split(df, column_name):
    y = df[column_name]  # Assign snow depth to y-axis (labels)
    X = df.drop([column_name, 'Date'], axis=1)  # Removes date and snow depth columns
    train_size = int(round(len(X) * 0.6, 0))  # Training data is 60% of data
    X_train, X_test, y_train, y_test = X.iloc[0:train_size], X.iloc[train_size + 1:-1], y.iloc[0:train_size], y.iloc[
                                                                                                  train_size + 1:-1]
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    return X_train, X_test, y_train, y_test


def scaling_values(X_train, X_test):  # Scaling the train values
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def plot_nn(predictions_s, y_test, predictions_n):
    X = np.linspace(0, 1, len(y_test))
    plt.figure(figsize=(8, 5))
    plt.plot(X, predictions_s, 'r-', label='MLP Shifted')
    plt.plot(X, predictions_n, 'g-', label='MLP Not Shifted')
    plt.plot(X, y_test, 'b-', label='Measured')
    plt.legend()
    plt.ylabel('Snow Depth (cm)')
    plt.xlabel('Date')
    plt.title('Fielding Lake Snow Depth')
    plt.show()


def MLP_fit(column_name, plot):
    file_name = 'D:/Users/Joshg/Documents/MDM3/Alaska/Fielding_Lake_1268_clean.csv'  # File location
    df = create_df(file_name, column_name)

    X_train_s, X_test_s, y_train, y_test = test_train_split(df, column_name)
    X_train_n, X_test_n = np.delete(X_train_s, 6, 1), np.delete(X_test_s, 6, 1)

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
        plot_nn(predictions_s, y_test, predictions_n)
    return rms_s, rms_n


# Using a for loop to calculate average rms
total_rms_s = 0  # Initialise a total counter
total_rms_n = 0
num_it = 2  # Number of iterations
for i in range(num_it):
    print('Iteration number ' + str(i))
    if i < num_it - 1:
        rms_s, rms_n = MLP_fit(column_name, 'no')  # Adding rms to mean total
        total_rms_s += rms_s
        total_rms_n += rms_n
    else:
        rms_s, rms_n = MLP_fit(column_name, 'yes')
        total_rms_s += rms_s
        total_rms_n += rms_n

print('--------- Using previous snow depth -------------')
print('average rms is ' + str(total_rms_s/num_it))  # Calculating and printing mean rms
print('------- Not using previous snow depth ------------')
print('average rms is ' + str(total_rms_n/num_it))  # Calculating and printing mean rms