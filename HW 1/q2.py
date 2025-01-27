import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import display
import matplotlib.pyplot as plt


# Load the data 
train = pd.read_csv('energy_train.csv')
test = pd.read_csv('energy_test.csv')
val = pd.read_csv('energy_val.csv')

'''
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

trainx, trainy = preprocess(train)
valx, valy = preprocess(test)
testx, testy = preprocess(val)

model = LinearRegression()
model.fit(trainx, trainy)

# create predictions
train_pred = model.predict(trainx)
val_pred = model.predict(valx)
test_pred = model.predict(testx)


'''
Apply the standard linear regression. Your must return following metrics and the associ-
ated values are the numeric values (a dictonary for example: {‘train-rmse’: 10.2, ‘train-r2’:
0.3, ‘val-rmse’: 7.2, ‘val-r2’: 0.2, ‘test-rmse’: 12.1, ‘test-r2’: 0.4}).
You ou can write a function eval linear1(trainx, trainy, valx, valy, testx, testy) that takes
in a training set, validation set, and test set, respectively, and trains a standard linear re-
gression model only on the training data and reports metrics on the training set, validation
set, and test set.
'''
def eval_linear1(trainx, trainy, valx, valy, testx, testy):
    model = LinearRegression()
    model.fit(trainx, trainy)
    train_pred = model.predict(trainx)
    val_pred = model.predict(valx)
    test_pred = model.predict(testx)
    train_rmse = np.sqrt(mean_squared_error(trainy, train_pred))
    val_rmse = np.sqrt(mean_squared_error(valy, val_pred))
    test_rmse = np.sqrt(mean_squared_error(testy, test_pred))
    train_r2 = r2_score(trainy, train_pred)
    val_r2 = r2_score(valy, val_pred)
    test_r2 = r2_score(testy, test_pred)
    return {'train-rmse': train_rmse, 'train-r2': train_r2, 'val-rmse': val_rmse, 'val-r2': val_r2, 'test-rmse': test_rmse, 'test-r2': test_r2}


output = eval_linear1(trainx, trainy, valx, valy, testx, testy)
print(output)


'''
Apply ridge. Write a function eval ridge(trainx, trainy, valx, valy, testx, testy, alpha) that
takes the regularization parameter, alpha, and trains a ridge regression model only on the
training data.
'''
def eval_ridge(trainx, trainy, valx, valy, testx, testy, alpha):
    model = Ridge(alpha=alpha)
    model.fit(trainx, trainy)
    train_pred = model.predict(trainx)
    val_pred = model.predict(valx)
    test_pred = model.predict(testx)
    return train_pred, val_pred, test_pred

'''
Apply lasso Write a function eval lasso(trainx, trainy, valx, valy, testx, testy, alpha) that
takes the regularization parameter, alpha, and trains a lasso regression model only on the
training data.
'''
def eval_lasso(trainx, trainy, valx, valy, testx, testy, alpha):
    model = Lasso(alpha=alpha)
    model.fit(trainx, trainy)
    train_pred = model.predict(trainx)
    val_pred = model.predict(valx)
    test_pred = model.predict(testx)
    return train_pred, val_pred, test_pred

'''
Report (using a table) the RMSE and R2 for training, validation, and test for all the
different λ values you tried. What would be the optimal parameter you would select based
on the validation data performance?
'''
def report(trainx, trainy, valx, valy, testx, testy, alphas):
    results = []
    for alpha in alphas:
        train_pred, val_pred, test_pred = eval_ridge(trainx, trainy, valx, valy, testx, testy, alpha)
        train_rmse = np.sqrt(mean_squared_error(trainy, train_pred))
        val_rmse = np.sqrt(mean_squared_error(valy, val_pred))
        test_rmse = np.sqrt(mean_squared_error(testy, test_pred))
        train_r2 = r2_score(trainy, train_pred)
        val_r2 = r2_score(valy, val_pred)
        test_r2 = r2_score(testy, test_pred)
        results.append((alpha, train_rmse, val_rmse, test_rmse, train_r2, val_r2, test_r2))
    df = pd.DataFrame(results, columns=['Alpha', 'Train RMSE', 'Val RMSE', 'Test RMSE', 'Train R2', 'Val R2', 'Test R2'])
    display(df)

alphas = [0.1, 1, 10, 100, 1000]
report(trainx, trainy, valx, valy, testx, testy, alphas)


'''
Generate the coefficient path plots (regularization value vs. coefficient value) for both ridge
and lasso. Make sure that your plots encompass all the expected behavior (coefficients
should shrink towards 0).
'''
def plot_coefficients(trainx, trainy, alpha, model):
    model = model(alpha=alpha)
    model.fit(trainx, trainy)
    plt.plot(model.coef_)
    plt.xlabel('Coefficient Index')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficient Path')
    plt.show()

plot_coefficients(trainx, trainy, 100, Ridge)

