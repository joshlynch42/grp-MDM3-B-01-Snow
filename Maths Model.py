import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from clean2 import clean_func

def auto_correlation_func(auto_days, noise):
    sum1 = 0
    for i in range(auto_days):
        sum1 += noise[auto_days:] * noise[i:i - auto_days]
    auto_correlation = np.sign(noise[auto_days:]) * sum1 / np.mean(auto_days * abs(noise))
    return auto_correlation

TEMP_AUTO_DAYS = 3
PREC_AUTO_DAYS = 3


start_dates = ['2014-11-01','2015-11-01','2016-11-01','2017-11-01','2018-11-01','2019-11-01','2020-11-01']
#start_dates = ['2014-11-01','2015-11-01']
years = len(start_dates)
t_all = np.array([])
snow_accumulation_all = np.array([])

for i in range(len(start_dates)-1):
    start_date = start_dates[i]
    end_date = start_dates[i+1]

    temp_mean, delta_temp, prec_mean, delta_prec, night_day_temp_diff, st, sp, tmax_list, tmin_list, actual_snow = clean_func(start_date, end_date)

    crit_temp = 0
    K = 3
    #M = N*(0.06*(tmax**2 - tmax*tmin + 0.18*tmax))

    N = 365
    t = np.linspace(i*365, (i+1)*365, N+1)

    tau = 365
    #st = 180

    temp_auto_days = TEMP_AUTO_DAYS
    temp_noise = np.random.normal(0, 5, N+1)
    temp_auto_days = int(temp_auto_days*N/(365))

    temp_auto_correlation_noise = auto_correlation_func(temp_auto_days, temp_noise)
    #temp = temp_mean + delta_temp*(np.sin(2*np.pi*(t-st)/tau)) + ((((-1)**(np.arange(N+1)%2==0))*night_day_temp_diff)/2)

    temp = temp_mean + delta_temp*(np.sin(2*np.pi*(t-st)/tau))
    auto_temp = temp[temp_auto_days:] + temp_auto_correlation_noise


    sp = 120
    #sp = 180

    prec_noise = np.random.normal(0, 5, N+1)
    prec_auto_days = PREC_AUTO_DAYS
    prec_auto_days = int(prec_auto_days*N/(365))

    prec_auto_correlation_noise = auto_correlation_func(prec_auto_days, prec_noise)

    #precipitation = prec_mean*(1 + delta_prec*(np.sin(2*np.pi*(t-sp)/tau)))
    precipitation = prec_mean * (1 + delta_prec * (np.sin(2 * np.pi * (t - sp) / tau)))/10
    auto_prec = precipitation[prec_auto_days:] + prec_auto_correlation_noise/10

    for i in range(len(auto_prec)):
        if auto_prec[i] < 0:
            auto_prec[i] = 0

    snow_accumulation = np.zeros(min(np.size(auto_temp), np.size(auto_prec)))
    h = (max(t) - min(t))/N

    for i in range(0, min(np.size(auto_temp), np.size(auto_prec))-1):
        if auto_temp[i] < crit_temp:
            x1 = auto_prec[i]
        elif snow_accumulation[i] <= 0:
            continue
        else:
            day_index = round(i*365/N)
            tmax = tmax_list[day_index]
            tmin = tmin_list[day_index]
            #K = h*(0.06*(tmax**2 - tmax*tmin + 0.18*tmax))
            K = h * (0.006 * (tmax ** 2 - tmax * tmin + 0.18 * tmax))
            x1 = -K*(temp[i]-crit_temp)

        snow_accumulation[i+1] = snow_accumulation[i] + x1*h
        if snow_accumulation[i+1] < 0:
            snow_accumulation[i+1] = 0

    t_all = np.append(t_all, t[max(temp_auto_days, prec_auto_days):])
    snow_accumulation_all = np.append(snow_accumulation_all, snow_accumulation)

temp_mean, delta_temp, prec_mean, delta_prec, night_day_temp_diff, st, sp, tmax_list, tmin_list, actual_snow = clean_func(start_dates[0], start_dates[-1])
t_snow = np.linspace(min(t_all),max(t_all),len(actual_snow))

rms = mean_squared_error(actual_snow[0:2178], snow_accumulation_all[0:2178], squared=False)
print(rms)

plt.plot(t_snow, actual_snow, label="Actual Snow")
plt.plot(t_all, snow_accumulation_all, label="Predicted Snow")
plt.title("Snow Accumulation vs Time")
plt.xlabel("Time (Days)")
plt.ylabel("Snow Accumulation (cm)")
plt.legend()
plt.grid()
plt.show()




