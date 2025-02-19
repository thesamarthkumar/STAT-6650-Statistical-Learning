import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Gini impurity calculation
def gini(y):
    _ , counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

# Find the best feature and threshold to split on
def find_best_split(xFeat, y):
    best_gain = 0
    best_feature = None
    best_threshold = None
    n_samples, n_features = xFeat.shape

    current_gini = gini(y)

    for feature in range(n_features):
        thresholds = np.unique(xFeat[:, feature])  # Try unique values as thresholds

        for threshold in thresholds:
            left_idx = xFeat[:, feature] <= threshold
            right_idx = xFeat[:, feature] > threshold

            if left_idx.sum() == 0 or right_idx.sum() == 0:
                continue  # Skip if a split results in empty nodes

            left_gini = gini(y[left_idx])
            right_gini = gini(y[right_idx])
            weighted_gini = (left_idx.sum() / n_samples) * left_gini + \
                            (right_idx.sum() / n_samples) * right_gini

            gain = current_gini - weighted_gini  # Information gain

            if gain > best_gain:
                best_gain, best_feature, best_threshold = gain, feature, threshold

    return best_feature, best_threshold

# Node structure
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction  # Only set for leaf nodes

# Train function
class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None
        self.depth = 0
        self.max_depth = 9
        self.min_samples = 5

    # (a) Define the train function using xFeat and y as parameters.
    def train(self, xFeat, y):
        # If all labels are the same, return a leaf node
        if len(np.unique(y)) == 1:
            return Node(prediction=y[0][0])

        # If stopping conditions are met, return a leaf node with the most common class
        if self.depth >= self.max_depth or len(y) < self.min_samples:
            most_common_label = np.bincount(y.flatten()).argmax()
            return Node(prediction=most_common_label)

        # Find the best feature and threshold to split the data
        best_feature, best_threshold = find_best_split(xFeat, y)

        # If no valid split is found, return a leaf node
        if best_feature is None:
            most_common_label = np.bincount(y.flatten()).argmax()
            return Node(prediction=most_common_label)

        # Split the data
        left_idx = xFeat[:, best_feature] <= best_threshold
        right_idx = xFeat[:, best_feature] > best_threshold

        # Recursively build left and right subtrees
        self.depth += 1
        left_subtree = self.train(xFeat[left_idx], y[left_idx])
        right_subtree = self.train(xFeat[right_idx], y[right_idx])

        # Return a node with the best feature, threshold, and subtrees
        self.tree = Node(feature=best_feature, threshold=best_threshold, 
                         left=left_subtree, right=right_subtree)
        
        
        return self.tree  # Important: Return the root node

    # (b) Defining the predict function.
    def _predict_single(self, row, node):
        """Traverse the tree to classify a single sample."""
        if node.prediction is not None:
            return node.prediction

        if row[node.feature] <= node.threshold:
            return self._predict_single(row, node.left)
        else:
            return self._predict_single(row, node.right)

    def predict(self, xTest):
        """Predict classes for multiple samples."""
        return np.array([self._predict_single(row, self.tree) for row in xTest]).reshape(-1, 1)

# Fetch dataset
wine_quality = fetch_ucirepo(id=186)

# Extract data
X = wine_quality.data.features  # Pandas DataFrame (features)
y = wine_quality.data.targets   # Pandas DataFrame (labels)

# Convert to NumPy
xFeat = X.to_numpy()
y = y.to_numpy()

# Convert quality scores into binary classification
y = np.where(y >= 6, 1, 0).reshape(-1, 1)  # Good quality (1) if ≥6, else Bad (0)


# Random seed
np.random.seed(42)
# Split into training (80%) and testing (20%)
indices = np.random.permutation(len(xFeat))
split_idx = int(0.8 * len(xFeat))
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

xTrain, yTrain = xFeat[train_idx], y[train_idx]
xTest, yTest = xFeat[test_idx], y[test_idx]

# Train the Decision Tree Model
dt = DecisionTreeClassifier()
dt.train(xTrain, yTrain)

# Predict on Test and Train Data
yPred = dt.predict(xTest)
yTrainPred = dt.predict(xTrain)

# Compute Test and Train Accuracy
accuracy = np.mean(yPred == yTest)
train_accuracy = np.mean(yTrainPred == yTrain)  
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

'''
Train Accuracy: 73.93%
Test Accuracy: 73.31%
'''

# Visualize the Decision Tree
def print_tree(node, depth=0):
    """Recursively prints the decision tree structure."""
    if node is None:
        return

    # Print leaf nodes
    if node.prediction is not None:
        print(f"{'  ' * depth}--> Predict: {node.prediction}")
        return

    # Print decision nodes
    print(f"{'  ' * depth}[Feature {node.feature} <= {node.threshold}]?")
    print(f"{'  ' * depth}--> True:")
    print_tree(node.left, depth + 1)
    print(f"{'  ' * depth}--> False:")
    print_tree(node.right, depth + 1)

# Call the function to print the trained tree
print_tree(dt.tree)

'''
Output:

[Feature 10 <= 10.1]?
--> True:
  [Feature 1 <= 0.27]?
  --> True:
    [Feature 1 <= 0.22]?
    --> True:
      [Feature 9 <= 0.48]?
      --> True:
        [Feature 3 <= 4.5]?
        --> True:
          [Feature 1 <= 0.18]?
          --> True:
            [Feature 5 <= 27.0]?
            --> True:
              [Feature 7 <= 0.99112]?
              --> True:
                --> Predict: 1
              --> False:
                [Feature 8 <= 3.21]?
                --> True:
                --> True:
                  --> Predict: 0
                --> False:
                --> True:
                  --> Predict: 0
                --> True:
                --> True:
                  --> Predict: 0
                --> True:
                --> True:
                --> True:
                --> True:
                --> True:
                --> True:
                --> True:
                --> True:
                --> True:
                --> True:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> True:
                --> True:
                --> True:
                  --> Predict: 0
                --> False:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> False:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> False:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> False:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> False:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> False:
                  --> Predict: 0
            --> False:
                --> True:
                  --> Predict: 0
                --> False:
                --> True:
                  --> Predict: 0
                --> False:
                  --> Predict: 0
            --> False:
              --> Predict: 1
                --> True:
                  --> Predict: 0
                --> False:
                  --> Predict: 0
            --> False:
                --> True:
                  --> Predict: 0
                --> False:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> False:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> False:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> False:
                  --> Predict: 0
            --> False:
                --> True:
                  --> Predict: 0
                --> False:
                --> True:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> True:
                  --> Predict: 0
                --> True:
                --> True:
                  --> Predict: 0
                  --> Predict: 0
                --> False:
                --> False:
                  --> Predict: 0
            --> False:
              --> Predict: 1
          --> False:
            --> Predict: 1
        --> False:
          --> Predict: 1
      --> False:
        --> Predict: 1
    --> False:
      --> Predict: 1
  --> False:
    --> Predict: 0
--> False:
  --> Predict: 1

'''
