import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt


def create_df(file_name, column_name):  # Creating a dataframe using a csv
    df = pd.read_csv(file_name)  # df stands for dataframe
    shifted = df[column_name]  # Creating an extra column containing the previous days snow depth
    shifted = shifted.shift(periods=1)
    df['Prev Snow Depth (cm)'] = shifted  # Adds previous snow levels to dataframe
    df = df.dropna()  # Removes null values
    return df


station_name = 'Fielding Lake (1268)'
column_name = station_name + ' Snow Depth (cm) Start of Day Values'
file_name = 'D:/Users/Joshg/Documents/MDM3/Alaska/Fielding_Lake_1268_clean.csv'  # File location
df = create_df(file_name, column_name)
df = df.dropna()
df.head()

sc1 = MinMaxScaler(feature_range = (0, 1))
Xs = sc1.fit_transform(df[['Alexander Lake (1267) Air Temperature Minimum (degC)', 'Alexander Lake (1267) Air Temperature Average (degC)']])
sc2 = MinMaxScaler(feature_range = (0, 1))
Ys = sc2.fit_transform(df[['Alexander Lake (1267) Air Temperature Average (degC)']])

# Each time step uses the previous window to predict the next value
window = 10

X_train = []
y_train = []
for i in range(window, len(Ys)):
    X_train.append(Xs[i-window:i,:])
    y_train.append(Ys[i])

X_train, y_train = np.array(X_train), np.array(y_train)

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 15, batch_size = 32)

y_predict = regressor.predict(X_train)

#unscale the output
y_predict_unscaled = sc2.inverse_transform(y_predict)
y_train_unscaled = sc2.inverse_transform(y_train)

X = np.linspace(0, 1, len(df[window:]))
plt.figure(figsize=(16,10))
plt.plot(X, y_predict_unscaled, 'r-', label='LSTM')
plt.plot(X, y_train_unscaled, 'b-', label='Measured')
plt.legend()
plt.ylabel('Temperature (C)')
plt.xlabel('Date')
plt.title('Alexander Lake Temperature')