import numpy as np 
import matplotlib.pyplot as plt    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt 

data = np.loadtxt("CombinedDriftTemp1.txt")
data[:,[0, 1]] = data[:,[1, 0]]
features = np.delete(data , 2 , 1)
labels = data[:,2]          #Original simulation data

kParamaterValuesList = []
for i in range(1 , 21):
    kParamaterValuesList.append(i)

train_scores, valid_scores = validation_curve(KNeighborsClassifier() , features , labels , param_name='n_neighbors' , param_range=kParamaterValuesList , cv=5)
#print(train_scores , "\n")
#print(valid_scores)

meanTrain_scores = np.mean(train_scores , axis=1)
meanValid_scores = np.mean(valid_scores , axis=1)
#print(meanTrain_scores)

plt.plot(kParamaterValuesList , meanTrain_scores , marker='.')
plt.plot(kParamaterValuesList , meanValid_scores , marker='x')

font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'bold',
        'size': 20,
        }

plt.tick_params(direction='inout', length=6, width=2, colors='black', labelsize=15)


plt.ylim((0 , 1))
plt.title("Validation Curve for Knn" , fontdict=font)
plt.xlabel("K(nearest neighbours)" , fontdict=font)
plt.ylabel("Accuracy" , fontdict=font)
plt.xticks([1 , 3 , 5 , 7 , 9 , 11 , 13 , 15 , 17 , 19 , 21])

plt.show()
