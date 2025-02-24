import time
import numpy as np
import pandas as pd
from IPython.display import display
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

# Convert quality scores into binary classification (Good quality (1) if ≥6, else Bad (0))
y = np.where(y >= 6, 1, 0)

'''
Initialize the Decision Tree Model.
'''
model = DecisionTreeClassifier(max_depth = 10, min_samples_split = 5)

'''
(a) Holdout Method
'''
def holdout(model, xFeat, y, test_size): 
    start = time.time()
    xTrain, xTest, yTrain, yTest = train_test_split(xFeat, y, test_size=test_size)
    model.fit(xTrain, yTrain)
    train_pred = model.predict_proba(xTrain)[:, -1]
    test_pred = model.predict_proba(xTest)[:, -1]
    train_auc = roc_auc_score(yTrain, train_pred)
    test_auc = roc_auc_score(yTest, test_pred)
    elapsed = time.time() - start
    return train_auc, test_auc, elapsed

# Call the holdout function, using test_size of 0.3 for a 70/30 split.
train_auc, test_auc, elapsed_time = holdout(model, xFeat, y, 0.3)
print(f'Holdout Method Results:')
print(f'Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, Elapsed Time: {elapsed_time:.2f} seconds \n')

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
    elapsed = time.time() - start
    return train_sum/k, test_sum/k, elapsed

# Call the kfold function, using k=5 for 5-fold CV.
train_auc, test_auc, elapsed_time = kfold(model, xFeat, y, 5)
print(f'K-Fold Cross-Validation Results:')
print(f'Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, Elapsed Time: {elapsed_time:.2f} seconds \n')

'''
(c) Monte Carlo Cross Validation
'''
def monte_carlo(model, xFeat, y, k):
    start = time.time()
    xTrain, xTest, yTrain, yTest = train_test_split(xFeat, y, test_size=0.7, random_state=np.random.randint(0,10000))
    model.fit(xTrain, yTrain)
    train_pred = model.predict_proba(xTrain)[:, -1]
    test_pred = model.predict_proba(xTest)[:, -1]
    train_auc = roc_auc_score(yTrain, train_pred)
    test_auc = roc_auc_score(yTest, test_pred)
    elapsed = time.time() - start
    return train_auc, test_auc, elapsed

# Call the monte_carlo function, using k=5 for 5-fold CV.
train_auc, test_auc, elapsed_time = monte_carlo(model, xFeat, y, 5)
print(f'Monte Carlo Cross-Validation Results:')
print(f'Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, Elapsed Time: {elapsed_time:.2f} seconds')
