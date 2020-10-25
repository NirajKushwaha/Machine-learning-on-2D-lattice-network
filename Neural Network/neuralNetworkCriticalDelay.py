import numpy as np 
import matplotlib.pyplot as plt    
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from statistics import mean

data = np.loadtxt("CombinedDriftTemp1.txt")
data[:,[0, 1]] = data[:,[1, 0]]
features = np.delete(data , 2 , 1)
labels = data[:,2]          #Original simulation data


numberOfModels = int(input("Enter the number of models you wanna create: "))
predictionMatrix = np.zeros((numberOfModels , 14))


for modelNumber in range(numberOfModels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    model = MLPClassifier(hidden_layer_sizes=(30,30) , activation='relu')
    model.fit(features , labels)

    predictingData = np.zeros((1,2))

    predictingData[0][0] = 0.001
    couplingCount = 0
    for coupling in np.arange(0.62 , 2.0 , 0.1):
        predictingData[0][1] = np.round(coupling , decimals=2)

        gate = 1        # 1 corresponds to open gate
        while(gate == 1):
            if(model.predict(predictingData) != 0):
                predictionMatrix[modelNumber][couplingCount] = predictingData[0][0]

                couplingCount += 1
                predictingData[0][0] = 0.1
                gate = 0    # Gate closed
            else:
                predictingData[0][0] = float(predictingData[0][0]+0.001)

criticalDelays = np.mean(predictionMatrix , axis=0)

print(criticalDelays)

font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'bold',
        'size': 20,
        }

plt.tick_params(direction='inout', length=6, width=2, colors='black', labelsize=19)



plt.plot(np.arange(0.62 , 2.0 , 0.1) , criticalDelays , marker='.')
plt.ylabel("Critical Delay" , fontdict=font)
plt.xlabel("Coupling Constant" , fontdict=font)
plt.xticks([0.75 , 1.25 , 1.75])

plt.show()