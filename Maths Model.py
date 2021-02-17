import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.integrate import odeint


def normal_distribution(x, mean, sd):
    prob_density = (1 / (2 * np.pi * sd ** 2)) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density

def func(x, t, temp, precipitation):
    x1 = x
    if temp < crit_temp:
        x1 = precipitation
    else:
        x1 = -K * (temp - crit_temp)

        # snow_accumulation[i + 1] = snow_accumulation[i] + x1 * h
        # if snow_accumulation[i + 1] <= 0:
        #     snow_accumulation[i + 1] = 0

    dxdt = x1
    return dxdt

#points = pd.read_csv('TestData.csv', header=None)

crit_temp = 0
K = 3
night_day_temp_diff = 10

#co-integration
#Euler–Maruyama

N = 100
t = np.linspace(1, 2*365, N+1)
temp_mean = 5
delta_temp = 15
tau = 365
st = 0

#temp = temp_mean + delta_temp*(np.sin(2*np.pi*(t-st)/tau))
temp = temp_mean + delta_temp*(np.sin(2*np.pi*(t-st)/tau)) + ((((-1)**(np.arange(N+1)%2==0))*night_day_temp_diff)/2) + np.random.normal(0, 5, N+1)


plt.plot(t, temp, label='Temp')

temp = np.sign(temp[:-1])*abs((temp[1:]*temp[:-1])/np.mean(abs(temp)))
plt.plot(t[:-1], temp, label='Autocorrelated Temp')

plt.title("Temperature vs Time")
plt.xlabel("Time (Days)")
plt.ylabel("Temperature (Degrees C)")
plt.legend()
plt.show()

prec_mean = 90
delta_prec = -5/9
sp = 0
#precipitation = prec_mean*(1 + delta_prec*(np.sin(2*np.pi*(t-sp)/tau)))
precipitation = prec_mean*(1 + delta_prec*(np.sin(2*np.pi*(t-sp)/tau))) + np.random.normal(0,5,N+1)

plt.plot(t, precipitation, label='Precipitation')

precipitation = np.sign(precipitation[:-1])*(precipitation[1:]*precipitation[:-1])/np.mean(abs(precipitation))

for i in range(len(precipitation)):
    if precipitation[i] < 0:
        precipitation[i] = 0

plt.plot(t[:-1], precipitation, label='Autocorrelated Precipitation')

plt.title("Precipitation vs Time")
plt.xlabel("Time (Days)")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.show()

# x0 = 0
# snow_accumulation = odeint(func, x0, t, args=(temp, precipitation))

snow_accumulation = np.zeros(np.shape(t))
h = (max(t) - min(t))/N

for i in range(0, N):
    if temp[i] < crit_temp:
        x1 = precipitation[i]
    elif snow_accumulation[i] <= 0:
        continue
    else:
        x1 = -K*(temp[i]-crit_temp)

    snow_accumulation[i+1] = snow_accumulation[i] + x1*h
    if snow_accumulation[i+1] < 0:
        snow_accumulation[i+1] = 0

plt.plot(t, snow_accumulation)
plt.title("Snow Accumulation vs Time")
plt.xlabel("Time (Days)")
plt.ylabel("Snow Accumulation (mm)")
plt.show()


