import numpy as np
from matplotlib import pyplot as plt, colors

train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)

norm = colors.Normalize(vmin=np.amin(train_GT), vmax=np.amax(train_GT))
fig, ax = plt.subplots()

ax.scatter(x=train_features[:,0], y=train_features[:,1], 
           color=plt.cm.hot(norm(train_GT)), s=1, alpha=.7)
sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=norm)
fig.colorbar(sm)
plt.show()