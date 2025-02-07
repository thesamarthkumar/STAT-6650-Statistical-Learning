import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Preprocessing the data.
'''
def preprocess(data):

    # Extract target variable
    X = data.drop(columns=['date', 'Appliances'], axis=1)
    Y = data['Appliances']

    # Convert to numpy
    X = X.to_numpy()
    Y = Y.to_numpy().reshape(-1, 1)  # Ensure trainy is 2D

    return X, Y

# Load the data
train = pd.read_csv('energy_train.csv')
test = pd.read_csv('energy_test.csv')
val = pd.read_csv('energy_val.csv')

# Preprocess the data
trainx, trainy = preprocess(train)
valx, valy = preprocess(test)
testx, testy = preprocess(val)

# Verify the shapes.
trainx.shape
trainy.shape

'''
Define the Ridge Regression Class.
'''
class RidgeSGD:

    # Default: lr=0.01, epochs=1000, batch_size=32, alpha=1.0, tol=1e-3
    def __init__(self, learning_rate, epochs, batch_size, alpha):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha  # Ridge regularization strength
        # self.tolerance = 1e-3
        self.weights = None
        self.bias = None

    # Predict y values.
    def predict(self, X):
        """Compute predictions"""
        return np.dot(X, self.weights) + self.bias

    # Compute MSE
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Gradient Descent 
    def gradient(self, X, y):
        y_pred = self.predict(X)
        
        # Ensure y is a 1D vector for correct broadcasting
        error = y_pred - y.flatten()

        # Compute gradient with Ridge regularization
        dW = (np.dot(X.T, error) / X.shape[0]) + (2 * self.alpha * self.weights)
        dB = np.mean(error)
        
        return dW, dB
        
    # Fit the model.
    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01  # Small random weights
        self.bias = np.random.randn()

        # Applying the Stochastic Gradient Descent (SGD)
        for epoch in range(self.epochs):

            # Shuffle the data.
            indices = np.random.permutation(n_samples)  # Shuffle data each epoch
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Compute the average gradient for each batch.
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                # Compute gradients
                gradient_weights, gradient_bias = self.gradient(X_batch, y_batch)

                # Gradient clipping (to prevent instability)
                clip_value = 1e3
                gradient_weights = np.clip(gradient_weights, -clip_value, clip_value)
                gradient_bias = np.clip(gradient_bias, -clip_value, clip_value)

                # Update weights and bias.
                self.weights -= self.learning_rate * gradient_weights
                self.bias -= self.learning_rate * gradient_bias

            # Print the loss at intervals of 100 epochs.
            if epoch % 100 == 0:
                y_pred = self.predict(X)
                loss = self.mean_squared_error(y, y_pred)
                print(f"Epoch {epoch}: Loss {loss:.4f}")

            # Convergence check
            # if np.linalg.norm(gradient_weights) < self.tolerance:
            #     print("Convergence reached.")
            #     break

        return self.weights, self.bias

'''
Run the model with the dataset.
'''
model = RidgeSGD(learning_rate=0.00001, epochs=1500, batch_size=1500, alpha=1.0)
model.fit(trainx, trainy)

# Create predictions
predictions = model.predict(testx)

# Print first few predictions
print(predictions[:5])

# Print the mean squared error
print(f'MSE: {model.mean_squared_error(testy, predictions)}')
