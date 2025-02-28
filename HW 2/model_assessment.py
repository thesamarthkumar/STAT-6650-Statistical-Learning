import time
import numpy as np
import pandas as pd
from IPython.display import display
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

#---------- Question 2 ----------#

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
Initialize the Decision Tree Model.
- max_depth: 10
- min_samples_split: 5
'''
model = DecisionTreeClassifier(max_depth = 10, min_samples_split = 5)

'''
(a) Holdout Method
'''
def holdout(model, xFeat, y, testSize): 
    start = time.time()
    xTrain, xTest, yTrain, yTest = train_test_split(xFeat, y, test_size=testSize)
    model.fit(xTrain, yTrain)
    trainAuc = roc_auc_score(yTrain, model.predict_proba(xTrain)[:, -1])
    testAuc = roc_auc_score(yTest, model.predict_proba(xTest)[:, -1])
    timeElapsed = time.time() - start
    return trainAuc, testAuc, timeElapsed

'''
(b) K-Fold Cross Validation
'''
def kfold(model, xFeat, y, k):
    trainSum, testSum = 0.0, 0.0  
    start = time.time()
    kf = KFold(n_splits=k, shuffle=True)  
    for train_index, test_index in kf.split(xFeat):
        xTrain, xTest = xFeat[train_index], xFeat[test_index]
        yTrain, yTest = y[train_index], y[test_index]
        model.fit(xTrain, yTrain)
        trainAuc = roc_auc_score(yTrain, model.predict_proba(xTrain)[:,-1])
        testAuc = roc_auc_score(yTest, model.predict_proba(xTest)[:,-1])
        trainSum += trainAuc
        testSum += testAuc
    timeElapsed = time.time() - start
    return trainSum/k, testSum/k, timeElapsed

'''
(c) Monte Carlo Cross Validation
'''
def monte_carlo(model, xFeat, y, testSize, s):
    trainSum, testSum = 0.0, 0.0
    start = time.time()
    for i in range(s):
        state = np.random.randint(0,10000)
        xTrain, xTest, yTrain, yTest = train_test_split(xFeat, y, test_size=testSize, random_state=state)
        model.fit(xTrain, yTrain)
        trainSum += roc_auc_score(yTrain, model.predict_proba(xTrain)[:, -1])
        testSum += roc_auc_score(yTest, model.predict_proba(xTest)[:, -1])
    timeElapsed = time.time() - start
    return trainSum/s, testSum/s, timeElapsed

'''
(d) For each method, retrieve the AUC values and the elapsed time.
'''
table = pd.DataFrame(columns=['trainAuc', 'testAuc', 'timeElapsed'])
table.loc['Holdout'] = holdout(model, xFeat, y, 0.3)
table.loc['K-Fold'] = kfold(model, xFeat, y, 10)
table.loc['Monte Carlo'] = monte_carlo(model, xFeat, y, 0.3, 50)

print('Results for each model assessment strategy:')
display(table)

'''
SAMPLE OUTPUT: 

Results for each model assessment strategy:
             trainAuc   testAuc  timeElapsed
Holdout      0.954645  0.766436     0.019680
K-Fold       0.945337  0.783827     0.227809
Monte Carlo  0.952188  0.770903     1.001136
'''
