import numpy as np 
import matplotlib.pyplot as plt    
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


data = np.loadtxt("CombinedDriftTemp1.txt")
data[:,[0, 1]] = data[:,[1, 0]]
features = np.delete(data , 2 , 1)
labels = data[:,2]          #Original simulation data

numberOfModels = int(input("Enter the number of models you wanna create: "))
numberOfDataPoints = np.size(labels)
predictionMatrix = np.zeros((numberOfModels , numberOfDataPoints))

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

print(predictionMatrix[:,2])

finalPredictionList = []

for i in range(numberOfDataPoints):
    predictionList = predictionMatrix[:,i]
    predictionList = predictionList.astype(int)
    bincountList = np.bincount(predictionList)

    finalPredictionList.append(bincountList.argmax())
    


confusionMatrix = confusion_matrix(labels , finalPredictionList)






def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix for SVM',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()




sensitivityList = []
for i in range(np.size(np.unique(labels))):
    rowSum = 0
    for j in range(np.size(np.unique(labels))):
        rowSum += confusionMatrix[i][j]

    sensitivity = confusionMatrix[i][i] / rowSum
    sensitivityList.append(sensitivity)

print(sensitivityList)    

specificityList = []
for k in range(np.size(np.unique(labels))):
    topSum = 0
    downColSum = 0
    for i in range(np.size(np.unique(labels))):
        for j in range(np.size(np.unique(labels))):
            if(i == k or j == k):
                pass
            else:
                topSum += confusionMatrix[i][j]

            if(i != j and j == k):
                downColSum += confusionMatrix[i][j]
            else:
                pass
    
    specificity = topSum / (topSum + downColSum)
    specificityList.append(specificity)

print(specificityList)




plot_confusion_matrix(cm=confusionMatrix , target_names=np.unique(labels) , normalize=False)