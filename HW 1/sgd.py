import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

class RidgeSGD:
    def __init__(self, learning_rate, regularization, epochs, batch_size):
        self.beta = None  # Coefficients will be initialized during training
        self.bias = 0  # Bias term
        self.learning_rate = learning_rate  # Learning rate (gamma)
        self.regularization = regularization  # Ridge regularization term (lambda)
        self.epochs = epochs  # Number of epochs
        self.batch_size = batch_size  # Batch size
        self.loss = None  # Stores final loss after training

    # Compute the gradient
    def gradient(self, X, y, y_pred):
        """
        Compute gradients for weights and bias.
        """
        err = y_pred - y
        dw = (1 / len(y)) * np.dot(X.T, err) + 2 * self.regularization * self.beta  # Ridge term
        db = (1 / len(y)) * np.sum(err)
        return dw, db

    # Evaluate the loss function
    def calculate_loss(self, y, y_pred):
        """
        Compute Mean Squared Error (MSE) loss.
        """
        mse_loss = (1 / (2 * len(y))) * np.sum((y_pred - y) ** 2)
        reg_loss = self.regularization * np.sum(self.beta ** 2)  # Ridge regularization term
        return mse_loss + reg_loss

    def train(self, X, y):
        """
        Train the Ridge Regression model using SGD.
        """
        n_samples, n_features = X.shape
        self.beta = np.random.randn(n_features) * 0.01  # Small random initialization
        self.bias = 0
        loss_history = {}

        for epoch in range(self.epochs):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # Predictions
                y_pred = np.dot(X_batch, self.beta) + self.bias

                # Prevent numerical instability
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    raise ValueError("NaN or Infinity detected in predictions!")

                # Compute gradients
                err = y_pred - y_batch
                dw = (1 / (len(y_batch) + 1e-8)) * np.dot(X_batch.T, err) + 2 * self.regularization * self.beta
                db = (1 / (len(y_batch) + 1e-8)) * np.sum(err)

                # Gradient clipping
                max_grad = 5
                dw = np.clip(dw, -max_grad, max_grad)
                db = np.clip(db, -max_grad, max_grad)

                # Update weights and bias
                self.beta -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                # Compute batch loss
                batch_loss = (1 / (2 * len(y_batch) + 1e-8)) * np.sum((y_pred - y_batch) ** 2) + self.regularization * np.sum(self.beta ** 2)
                epoch_loss += batch_loss * len(y_batch)

            # Store average loss for the epoch
            loss_history[epoch] = epoch_loss / n_samples

            # Check for NaN in weights or loss
            if np.isnan(loss_history[epoch]) or np.any(np.isnan(self.beta)):
                raise ValueError("NaN encountered during training. Check learning rate or data scaling.")

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss_history[epoch]:.4f}")

        self.loss = loss_history
        return loss_history


    # Predict using the trained model
    def predict(self, X):
        """
        Predict target values for input data.
        """
        return np.dot(X, self.beta) + self.bias

'''
2b)
Function to preprocess the data.
'''
def preprocess(data):

    '''
    Move the Appliances column (target variable) to the end of the dataframe.
    Separate the features (X) from the target variable (Y).
    '''
    Y = data['Appliances']
    data.drop(['Appliances'], axis=1, inplace=True)
    X = data.iloc[:, :-1]

    '''
    Fix the date formatting (example: 1/11/16 17:00)
    Replace the date column with new columns:
    - day
    - month
    - year
    - hour
    - minute
    '''
    X['date'] = pd.to_datetime(X['date'], format='%m/%d/%y %H:%M')
    X['day'] = X['date'].dt.day
    X['month'] = X['date'].dt.month
    X['year'] = X['date'].dt.year
    X['hour'] = X['date'].dt.hour
    X['minute'] = X['date'].dt.minute
    X.drop(['date'], axis=1, inplace=True)

    # Return the X and Y values.
    return X, Y

def main():
    import pandas as pd

    # Load the data
    train = pd.read_csv('energy_train.csv')
    test = pd.read_csv('energy_test.csv')
    val = pd.read_csv('energy_val.csv')

    # Preprocess the data
    trainx, trainy = preprocess(train)
    valx, valy = preprocess(val)
    testx, testy = preprocess(test)

    # Align feature columns across datasets
    common_columns = trainx.columns.intersection(valx.columns).intersection(testx.columns)
    trainx = trainx[common_columns]
    valx = valx[common_columns]
    testx = testx[common_columns]

    # Optional: Handle missing values
    trainx.fillna(0, inplace=True)
    valx.fillna(0, inplace=True)
    testx.fillna(0, inplace=True)

    # At this point, trainx, valx, and testx are ready for training and evaluation
    print("Preprocessing complete!")
    print(f"Train shape: {trainx.shape}, Val shape: {valx.shape}, Test shape: {testx.shape}")

    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    # Instantiate and train the Ridge Regression model using SGD
    ridge_sgd = RidgeSGD(learning_rate=0.01, regularization=0.1, epochs=100, batch_size=32)
    loss_history = ridge_sgd.train(trainx.values, trainy.values)

    # Predict on training, validation, and test sets
    train_pred = ridge_sgd.predict(trainx.values)
    val_pred = ridge_sgd.predict(valx.values)
    test_pred = ridge_sgd.predict(testx.values)

    # Calculate metrics for train, validation, and test sets
    train_rmse = np.sqrt(mean_squared_error(trainy, train_pred))
    val_rmse = np.sqrt(mean_squared_error(valy, val_pred))
    test_rmse = np.sqrt(mean_squared_error(testy, test_pred))

    train_r2 = r2_score(trainy, train_pred)
    val_r2 = r2_score(valy, val_pred)
    test_r2 = r2_score(testy, test_pred)

    # Print results
    print(f"Train RMSE: {train_rmse:.2f}, R^2: {train_r2:.2f}")
    #print(f"Validation RMSE: {val_rmse:.2f}, R^2: {val_r2:.2f}")
    #print(f"Test RMSE: {test_rmse:.2f}, R^2: {test_r2:.2f}")


if __name__ == "__main__":
    main()

        

'''
Execute the main function.
'''
if __name__ == '__main__':
    main()
