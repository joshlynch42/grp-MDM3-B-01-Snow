import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get the dataset
df = pd.read_csv('C:/Users/ale/Documents/Code/MDM3_Snow/Alaska/Alexander_Lake_1267_clean.csv')
X = df.loc[:, ['Alexander Lake (1267) Air Temperature Minimum (degC)', 'Alexander Lake (1267) Air Temperature Average (degC)', 'Alexander Lake (1267) Air Temperature Maximum (degC)', 'Alexander Lake (1267) Precipitation Accumulation (mm) Start of Day Values', 'Alexander Lake (1267) Precipitation Increment - Snow-adj (mm)', 'Alexander Lake (1267) Snow Depth (cm) Start of Day Values', 'Alexander Lake (1267) Snow Density (pct) Start of Day Values']]
X = X.to_numpy()
X = np.transpose(X)
#calculate covariance matrix
cov_data = np.corrcoef(X)
#plot a pretty picture
img = plt.matshow(cov_data, cmap=plt.cm.rainbow)
plt.colorbar(img, ticks=[-1, 0, 1], fraction=0.045)
for x in range(cov_data.shape[0]):
    for y in range(cov_data.shape[1]):
        plt.text(x, y, "%0.2f" % cov_data[x, y], size=12, color='black', ha="center", va="center")
plt.show()
