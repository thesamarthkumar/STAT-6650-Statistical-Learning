import time
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

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
'''
model = DecisionTreeClassifier(max_depth = 10, min_samples_split = 5)

'''
(a) Holdout Method
'''
def holdout(model, xFeat, y, testSize): 
    start = time.time()
    xTrain, xTest, yTrain, yTest = train_test_split(xFeat, y, test_size=testSize)
    model.fit(xTrain, yTrain)
    train_pred = model.predict_proba(xTrain)[:, -1]
    test_pred = model.predict_proba(xTest)[:, -1]
    train_auc = roc_auc_score(yTrain, train_pred)
    test_auc = roc_auc_score(yTest, test_pred)
    timeElapsed = time.time() - start
    return train_auc, test_auc, timeElapsed

# Call the holdout function, using test_size of 0.3 for a 70/30 split.
train_auc, test_auc, timeElapsed = holdout(model, xFeat, y, 0.3)
print(f'Holdout Method Results:')
print(f'Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, Elapsed Time: {timeElapsed:.2f} seconds \n')

'''
(b) K-Fold Cross Validation
'''
def kfold(model, xFeat, y, k):
    start = time.time()
    kf = KFold(n_splits=k, shuffle=True)
    train_sum, test_sum = 0.0, 0.0    
    for train_index, test_index in kf.split(xFeat):
        X_train, X_test = xFeat[train_index], xFeat[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:,-1])
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,-1])
        train_sum += train_auc
        test_sum += test_auc
    timeElapsed = time.time() - start
    return train_sum/k, test_sum/k, timeElapsed

# Call the kfold function, using k=10 for 10-fold CV.
train_auc, test_auc, timeElapsed = kfold(model, xFeat, y, 10)
print(f'K-Fold Cross-Validation Results:')
print(f'Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, Elapsed Time: {timeElapsed:.2f} seconds \n')

'''
(c) Monte Carlo Cross Validation
'''
def monte_carlo(model, xFeat, y, testSize, s):
    train_sum, test_sum = 0.0, 0.0
    start = time.time()
    for i in range(s):
        rand_state = np.random.randint(0,10000)
        xTrain, xTest, yTrain, yTest = train_test_split(xFeat, y, test_size=testSize, random_state=rand_state)
        model.fit(xTrain, yTrain)
        train_pred = model.predict_proba(xTrain)[:, -1]
        test_pred = model.predict_proba(xTest)[:, -1]
        train_sum += roc_auc_score(yTrain, train_pred)
        test_sum += roc_auc_score(yTest, test_pred)
    timeElapsed = time.time() - start
    return train_sum/s, test_sum/s, timeElapsed

# Call the monte_carlo function, using s = 50 iterations and testSize of 0.3 for a 70/30 split.
train_auc, test_auc, timeElapsed = monte_carlo(model, xFeat, y, 0.3, 50)
print(f'Monte Carlo Cross-Validation Results:')
print(f'Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, Elapsed Time: {timeElapsed:.2f} seconds')

'''
RESULTS:
Holdout Method Results:
Train AUC: 0.9543, Test AUC: 0.7525, Elapsed Time: 0.02 seconds 

K-Fold Cross-Validation Results:
Train AUC: 0.9461, Test AUC: 0.7855, Elapsed Time: 0.27 seconds 

Monte Carlo Cross-Validation Results:
Train AUC: 0.9525, Test AUC: 0.7747, Elapsed Time: 0.95 seconds
'''

'''
TO DO:
- Complete part (d).
- Formatting. Consistency throughout the entire code.
'''
