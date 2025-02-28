import time
import numpy as np
import pandas as pd
from IPython.display import display
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


#---------- Question 3 ----------#

'''
Load the Data.
'''
# Load the Wine Quality dataset from UCI ML Repository.
wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features  
y = wine_quality.data.targets  

# xFeat is an n x d array.
xFeat = X.to_numpy()

# y is an n x 1 array.
y = y.to_numpy().ravel()

# Convert quality scores into binary classification.
# Good Quality (1) when y is at least 6.
# Bad Quality (0) when 0 < y <= 5.
y = np.where(y >= 6, 1, 0)


'''
Initialize Decision Tree and KNN Models.
'''
dtModel = DecisionTreeClassifier()
knnModel = KNeighborsClassifier()


'''
(a) Use K-Fold CV to find optimal parameters for each Decision Tree and KNN.
'''

# Decision Tree
dtMaxDepth = [5, 10, 15, 20]
dtMinSamplesSplit = [2, 5, 10, 15]
dtOptimalParams = (0, 0)
dtOptimalAuc = 0.0

for maxDepth in dtMaxDepth:
    for minSamplesSplit in dtMinSamplesSplit:
        model = DecisionTreeClassifier(max_depth=maxDepth, min_samples_split=minSamplesSplit)
        trainSum, testSum = 0.0, 0.0  
        kf = KFold(n_splits=5, shuffle=True)  
        for train_index, test_index in kf.split(xFeat):
            xTrain, xTest = xFeat[train_index], xFeat[test_index]
            yTrain, yTest = y[train_index], y[test_index]
            model.fit(xTrain, yTrain)
            trainSum += roc_auc_score(yTrain, model.predict_proba(xTrain)[:, -1])
            testSum += roc_auc_score(yTest, model.predict_proba(xTest)[:, -1])
        trainAuc = trainSum / 5
        testAuc = testSum / 5
        if testAuc > dtOptimalAuc:
            dtOptimalAuc = testAuc
            dtOptimalParams = (maxDepth, minSamplesSplit)

# KNN
knnNeighbors = [5, 10, 15, 20]
knnOptimalParams = 0
knnOptimalAuc = 0.0

for neighbors in knnNeighbors:
    model = KNeighborsClassifier(n_neighbors=neighbors)
    trainSum, testSum = 0.0, 0.0  
    kf = KFold(n_splits=5, shuffle=True)  
    for train_index, test_index in kf.split(xFeat):
        xTrain, xTest = xFeat[train_index], xFeat[test_index]
        yTrain, yTest = y[train_index], y[test_index]
        model.fit(xTrain, yTrain)
        trainSum += roc_auc_score(yTrain, model.predict_proba(xTrain)[:, -1])
        testSum += roc_auc_score(yTest, model.predict_proba(xTest)[:, -1])
    trainAuc = trainSum / 5
    testAuc = testSum / 5
    if testAuc > knnOptimalAuc:
        knnOptimalAuc = testAuc
        knnOptimalParams = neighbors

display(pd.DataFrame({
    'Decision Tree': [dtOptimalParams, dtOptimalAuc],
    'KNN': [knnOptimalParams, knnOptimalAuc]
}, index=['Optimal Parameters', 'Optimal AUC']))

print('\n')

'''
(b) KNN Model.
- use optimal parameters from (a)
- train on entire dataset
- evaluate AUC and Accuracy on test dataset
- create 3 datasets where you randomly remove 1%, 5%, 10% of the original training data
- train KNN on each subset
- evaluate AUC and Accuracy on test dataset
'''

# Train KNN on entire dataset.
knnModel = KNeighborsClassifier(n_neighbors=knnOptimalParams)
knnModel.fit(xFeat, y)

# Evaluate AUC and Accuracy on test dataset.
testAuc_knn = roc_auc_score(y, knnModel.predict_proba(xFeat)[:, -1])
testAccuracy_knn = knnModel.score(xFeat, y)

# Create 3 datasets where you randomly remove 1%, 5%, 10% of the original training data.
randomStates = [1, 2, 3]
percentages = [0.01, 0.05, 0.10]

results = []
results.append(['KNN Entire', testAuc_knn, testAccuracy_knn])

for percent in percentages:
    # KNN Model on each subset.
    knnModel = KNeighborsClassifier(n_neighbors=knnOptimalParams)
    xTrain, xTest, yTrain, yTest = train_test_split(xFeat, y, test_size=percent)
    knnModel.fit(xTrain, yTrain)
    testAuc = roc_auc_score(yTest, knnModel.predict_proba(xTest)[:, -1])
    testAccuracy = knnModel.score(xTest, yTest)
    results.append([f'KNN ({percent*100}%)', testAuc, testAccuracy])

'''
(c) Decision Tree Model.
- use optimal parameters from (a)
- train on entire dataset
- evaluate AUC and Accuracy on test dataset
- create 3 datasets where you randomly remove 1%, 5%, 10% of the original training data
- train Decision Tree on each subset
- evaluate AUC and Accuracy on test dataset
'''

# Train Decision Tree on entire dataset.
dtModel = DecisionTreeClassifier(max_depth=dtOptimalParams[0], min_samples_split=dtOptimalParams[1])
dtModel.fit(xFeat, y)

# Evaluate AUC and Accuracy on test dataset.
testAuc_tree = roc_auc_score(y, dtModel.predict_proba(xFeat)[:, -1])
testAccuracy_tree = dtModel.score(xFeat, y)
results.append(['Decision Tree Entire', testAuc_tree, testAccuracy_tree])

for percent in percentages:
    dtModel = DecisionTreeClassifier(max_depth=dtOptimalParams[0], min_samples_split=dtOptimalParams[1])
    xTrain, xTest, yTrain, yTest = train_test_split(xFeat, y, test_size=percent)
    dtModel.fit(xTrain, yTrain)
    testAuc = roc_auc_score(yTest, dtModel.predict_proba(xTest)[:, -1])
    testAccuracy = dtModel.score(xTest, yTest)
    results.append([f'Decision Tree ({percent*100}%)', testAuc, testAccuracy])


'''
(d) Display the results for all 8 models.
'''
display(pd.DataFrame(results, columns=['Model', 'Test AUC', 'Test Accuracy']))

'''
SAMPLE OUTPUT: 
                   Decision Tree        KNN
Optimal Parameters      (10, 15)  10.000000
Optimal AUC             0.796011   0.710536


                   Model  Test AUC  Test Accuracy
0             KNN Entire  0.812229       0.742650
1             KNN (1.0%)  0.699275       0.676923
2             KNN (5.0%)  0.676371       0.636923
3            KNN (10.0%)  0.701864       0.686154
4   Decision Tree Entire  0.929244       0.857627
5   Decision Tree (1.0%)  0.798097       0.753846
6   Decision Tree (5.0%)  0.777594       0.747692
7  Decision Tree (10.0%)  0.802049       0.749231

'''
