import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#load in data
S1 = pd.read_csv('Alaska/Alexander_Lake_1267.csv')
S2 = pd.read_csv('Alaska/Fielding_Lake_1268.csv')
S3 = pd.read_csv('Alaska/Flower_Mountain_1285.csv')
S4 = pd.read_csv('Alaska/Heen_Latinee_1270.csv')
S5 = pd.read_csv('Alaska/Jack_Wade_Jct_1275.csv')
#create a list of dates: replace start and end values with the range of the data at the station
time_axis = pd.date_range(start='10/01/2014', end='02/03/2021')
time_axis = pd.Series(time_axis)
#remove date column since not needed
station_no_date = S1.drop(['Date'], axis=1)

Data = []
#setting up indexes to remove nans
for row, column in station_no_date.iteritems():
    column = pd.DataFrame(column)
    index = column.isnull()
    index = index.astype(int)
    index = index.to_numpy()
    index = index.tolist()
    column_index = column.index.values.tolist()
    index = [column_index.index(i[0]) for i in index]
    value = column.to_numpy()
    value = value.tolist()
    values = []
    for i in value:
        values += i

    inds = []
    counter = 0
    for i in index:
        if i == 1:
            inds.append(counter)
        counter += 1
#remove nans
    def nanSmoother():
        temp = []
        prev_found_value = 'nan'
        inds_index = 0

        for i in inds:
            if i == len(values)-1:
                temp.append(i)
                for stored_indexes in temp:
                    values[stored_indexes] = prev_found_value

            elif np.isnan(values[i+1]):
                temp.append(i)

            else:

                temp.append(i)
                for stored_indexes in temp:
                    if prev_found_value == 'nan':
                        values[stored_indexes] = values[i+1]
                    else:
                        values[stored_indexes] = np.mean([values[i+1], prev_found_value])

                temp = []
                try:
                    prev_found_value = values[inds[inds_index+1]-1]
                except IndexError:
                    print('reached end of data')
            inds_index += 1
    nanSmoother()
    Data.append(values)

Data = np.transpose(Data)
final_df = pd.DataFrame(Data)
final_df = final_df.set_index(time_axis)
Moving_average = final_df.rolling(31).mean()
#print in blue the uncleaned data, in red the cleaned data
#change the date values to the range you wanna view
plot_df = Moving_average.loc['2015-01-01':'2020-01-01']
plot_df.plot(kind='line', use_index=True, y=5, color='red', label='Alexander Lake (1267) Snow Depth (cm) Start of Day Values')
plot_df = final_df.loc['2015-01-01':'2020-01-01']
plot_df.plot(kind='line', use_index=True, y=5, color='blue')
plt.show()
