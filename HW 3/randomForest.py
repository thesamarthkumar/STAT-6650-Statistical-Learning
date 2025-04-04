import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder 

'''
Using the DecisionTreeClassifier from Scikit-learn with a custom implementation of a pseudo Random Forest.
'''
class RandomForest:

    def __init__(self, nest=10, maxDepth=None, minSamplesLeaf=1, q=None, randomState=None):
        self.nest = nest
        self.maxDepth = maxDepth
        self.minSamplesLeaf = minSamplesLeaf
        self.q = q
        self.randomState = randomState
        self.trees = []
        self.featureSubsets = []

    # Fit the model with training data.
    def fit(self, X, y):
        # Seed the random generator if random_state is provided
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
        numSamples, numFeatures = X.shape
        
        # If q is not provided, default to int(sqrt(numFeatures))
        if self.q is None:
            self.q = int(np.sqrt(numFeatures))
        
        for _ in range(self.nest):
            # Choose a random subset of features (subspace)
            featureIdx = np.random.choice(numFeatures, self.q, replace=False)
            
            # Bootstrap the data rows
            sampleIdx = np.random.choice(numSamples, numSamples, replace=True)
            
            # Subset X and y based on bootstrap sample and feature subspace
            xBoot = X[sampleIdx][:, featureIdx]
            yBoot = y[sampleIdx]
            
            # Fit a decision tree to the bootstrapped data.
            tree = DecisionTreeClassifier(
                max_depth=self.maxDepth, 
                min_samples_leaf=self.minSamplesLeaf,
                random_state=self.randomState
            )
            tree.fit(xBoot, yBoot)
            
            # Save the fitted tree and feature subset.
            self.trees.append(tree)
            self.featureSubsets.append(featureIdx)

    # Create predictions.
    def predict(self, X):
        # Collect predictions from each tree
        predictions = []
        for i, tree in enumerate(self.trees):
            # Grab only the feature subspace used by tree i
            featureIdx = self.featureSubsets[i]
            X_sub = X[:, featureIdx]
            
            # Predict using this tree
            preds = tree.predict(X_sub)
            predictions.append(preds)
        
        # Transpose predictions so each row corresponds to one sample across all trees
        predictions = np.array(predictions).T
        
        # Majority vote for each sample
        finalPredictions = []
        for row in predictions:
            vote = np.bincount(row).argmax()
            finalPredictions.append(vote)
        
        return np.array(finalPredictions)
      
# Load and preprocess the data, return the test/train split.
def preprocess(fileName):
    df = pd.read_csv(fileName)

    df.drop('ID', axis=1, inplace=True)

    # Handle missing values.
    for col in df.columns:
        # Using the mode for categorical features.
        if df[col].dtype == 'object':
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
        # Mean for numerical features.
        else:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)

    # Using label encoding so that we use integer values for categorical features.
    cat_col = [c for c in df.columns if df[c].dtype == 'object']
    for c in cat_col:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])

    # Split features & target
    X = df.drop('Status', axis=1).values
    y = df['Status'].values

    # Split training and test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Return the split data.
    return X_train, X_test, y_train, y_test
  
# Training the RandomForest classifier with the dataset.
if __name__ == "__main__":

    # Load and preprocess the dataset.
    X_train, X_test, y_train, y_test = preprocess("Loan_Default.csv")

    # Initialize the RandomForest model.
    model = RandomForest(
        nest=5,
        maxDepth=5,
        minSamplesLeaf=10,
        q=8,
        randomState=42
    )

    # Train the model.
    model.fit(X_train, y_train)
    
    # Predict on the test set
    testPred = model.predict(X_test)
    
    # Evaluate the model
    acc = accuracy_score(y_test, testPred)
    print("Accuracy on test set:", acc)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, testPred))
    print("Classification Report:")
    print(classification_report(y_test, testPred))


'''
Sample Output:

Accuracy on test set: 0.9991928432097935
Confusion Matrix:
[[22486     8]
 [   16  7224]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     22494
           1       1.00      1.00      1.00      7240

    accuracy                           1.00     29734
   macro avg       1.00      1.00      1.00     29734
weighted avg       1.00      1.00      1.00     29734

'''
