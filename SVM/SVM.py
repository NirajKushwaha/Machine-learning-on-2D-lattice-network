import numpy as np 
import matplotlib.pyplot as plt    
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from statistics import mean

data = np.loadtxt("CombinedDriftTemp1.txt")
data[:,[0, 1]] = data[:,[1, 0]]
features = np.delete(data , 2 , 1)
labels = data[:,2]          #Original simulation data

numberOfModels = int(input("Enter the number of models you wanna create: "))
numberOfDataPoints = np.size(labels)
predictionMatrix = np.zeros((numberOfModels , numberOfDataPoints))

testAccuracyList = []
trainAccuracyList = []

for modelNumber in range(numberOfModels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    model = SVC(kernel='rbf' , C=10 , gamma=0.5)
    model.fit(X_train , y_train)
    



    #x1 = np.arange(0 , 1 , 0.025)
    #x2 = np.arange(1 , 20.5 , 0.5)
    x1 = np.arange(0 , 1 , 0.05)
    x2 = np.arange(1 , 21 , 1)
    xFinal = np.concatenate((x1 , x2) , axis=None)

    #yFinal = np.arange(0.5 , 2.0375 , 0.0375)
    yFinal = np.arange(0.5 , 2 , 0.075)

    x , y = np.meshgrid(xFinal , yFinal)
    size = x.shape

    temp = np.zeros((xFinal.size*yFinal.size , 2))

    a = 0
    for i in yFinal:
        for j in xFinal:
            temp[a][0] = j
            temp[a][1] = i

            a += 1

    predictionList = model.predict(temp)
    
    predictionMatrix[modelNumber,:] = predictionList



        #<<=====================For calculating Test and Train accuracy=====================>>
    i = 0
    correct = 0
    wrong = 0
    testPrediction = model.predict(X_test)
    for x in y_test:
        if(x == testPrediction[i]):
            correct += 1
        else:
            wrong += 1

        i += 1
    testAccuracyList.append(correct/(correct+wrong))

    i = 0
    correct = 0
    wrong = 0
    trainPrediction = model.predict(X_train)
    for x in y_train:
        if(x == trainPrediction[i]):
            correct += 1
        else:
            wrong += 1

        i += 1
    trainAccuracyList.append(correct/(correct+wrong))
    #<<=====================For calculating Test and Train accuracy=====================>>

print("Testing Accuracy:" , mean(testAccuracyList)*100)
print("Training Accuracy:" , mean(trainAccuracyList)*100)



finalPredictionList = []

for i in range(numberOfDataPoints):
    predictionList = predictionMatrix[:,i]
    predictionList = predictionList.astype(int)
    bincountList = np.bincount(predictionList)

    finalPredictionList.append(bincountList.argmax())
    


i = 0
correct = 0
wrong = 0
for x in finalPredictionList:
    if(x == labels[i]):
        correct += 1
    else:
        wrong += 1
    
    i += 1

print((correct/(correct+wrong))*100 , "Percent accuracy obtained")


#x1 = np.arange(0 , 1 , 0.025)
#x2 = np.arange(1 , 20.5 , 0.5)
x1 = np.arange(0 , 1 , 0.05)
x2 = np.arange(1 , 21 , 1)
xFinal = np.concatenate((x1 , x2) , axis=None)

#yFinal = np.arange(0.5 , 2.0375 , 0.0375)
yFinal = np.arange(0.5 , 2 , 0.075)

x , y = np.meshgrid(xFinal , yFinal)
size = x.shape

z = np.reshape(finalPredictionList , size)
#print(z)

figure = plt.contourf(x , y , z , levels=3)
#plt.pcolormesh(z)


#np.savetxt("zSVM.txt" , z)



font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'bold',
        'size': 20,
        }

plt.tick_params(direction='inout', length=6, width=2, colors='black', labelsize=19)


plt.title("SVM(rbf)" , fontdict=font)
#plt.xlim(0 , 11)
plt.yticks([0.5 , 1.0 , 1.5 , 1.9])
plt.xlabel("Delay" , fontdict=font)
plt.ylabel("Coupling Constant" , fontdict=font)

bar = plt.colorbar(figure , ticks=[3 , 9 , 15 , 21])
bar.ax.set_yticklabels(['0', '9', '13' , '21'])
bar.set_label(label="Number of Drifiting Oscillators" , weight='bold' , size=13)

plt.show()

    
