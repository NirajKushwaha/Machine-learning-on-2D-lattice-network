import numpy as np    
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from statistics import mean
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

data = np.loadtxt("CombinedDriftTemp1.txt")
features = np.delete(data , 2 , 1)
labels = data[:,2]          #Original simulation data

numberOfModels = 1     
numberOfDataPoints = np.size(labels)
predictionMatrix = np.zeros((numberOfModels , numberOfDataPoints))

testAccuracyList = []
trainAccuracyList = []


for modelNumber in range(numberOfModels):
    X, X_test, y, y_test = train_test_split(features, labels, test_size=0.2)

    clf = MLPClassifier(hidden_layer_sizes=(30,30) , activation='relu')
    clf.fit(X , y)

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
#plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors, weights))

plt.show()


