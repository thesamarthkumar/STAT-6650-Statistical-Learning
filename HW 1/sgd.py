import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


'''
Preprocess the  (Same as Q2).
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

'''
Define the ridge regression class.
'''

class RidgeSGD:

    def __init__(self, beta, alpha, epochs, batch_size):
        self.beta = beta
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = None
        pass

    # Evaluate the loss function.
    def loss_function(self, x, y):
        


        pass

    # Compute the gradient.
    def gradient(self, X, y, y_pred):
        err = y_pred - y
        dw = (1/len(y)) * np.dot(X.T, err)
        db = (1/len(y)) * np.sum(err)
        return dw, db

    # Define the train function that trains the ridge regression model using stochastic gradient descent.
    # The train function returns a dictionary where the key denotes the epoch number 
    # and the value denotes the loss associated with that epoch.
    def train(self, x, y):
        pass

    def predict(self, X):
        return np.dot(X, self.beta)
    

    '''
    sgd_algorithm():
        for epoch in range(epochs):
            shuffle the training data and break into B = N/BatchSize batches.
            for b in range(B):
                # Compute the gradients associated with samples in batch b, upside down delta F
                # Take the average of the gradients in the batch.
                # Update the parameters based on the weight.
    '''

    def sgd_algorithm(self, x, y):
        
        n_samples = x.shape[0]
        n_features = x.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        loss_history = {}

        for epoch in range(self.epochs):

            # shuffle the data
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            for i in range(0, n_samples, self.batch_size):
                x_i = x_shuffled[i:i+self.batch_size]
                y_i = y_shuffled[i:i+self.batch_size]

                y_pred = np.dot(x_i, self.weights) + self.bias
                dw, db = self.gradient(x_i, y_i, y_pred)

                self.weights -= self.alpha * dw
                self.bias -= self.alpha * db

                batch_loss = (1 / (2 * len(y_i))) * np.sum((y_pred - y_i) ** 2)
                epoch_loss += batch_loss

            loss_history[epoch] = epoch_loss / (n_samples // self.batch_size)
        
        
        pass



'''
Main function to run the code.
'''

def main():
    # Load the data
    train = pd.read_csv('energy_train.csv')
    test = pd.read_csv('energy_test.csv')
    val = pd.read_csv('energy_val.csv')

    # Preprocess the data
    trainx, trainy = preprocess(train)
    valx, valy = preprocess(test)
    testx, testy = preprocess(val)

    

'''
Execute the main function.
'''
if __name__ == '__main__':
    main()
