import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

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

    # Evaluate the linear regression model for train, test, and validation data.
    output_dict = eval_linear1(trainx, trainy, valx, valy, testx, testy)
    print(f'2b) Eval Linear: \n{output_dict}')

    # Apply ridge with alpha = 0.1
    train_ridge, val_ridge, test_ridge = eval_ridge(trainx, trainy, valx, valy, testx, testy, 0.1)

    # Apply lasso with alpha = 0.1
    train_lasso, val_lasso, test_lasso = eval_lasso(trainx, trainy, valx, valy, testx, testy, 0.1)

    # Reporting results with a table using different alpha values
    report(trainx, trainy, valx, valy, testx, testy)

    # Plotting coefficients for ridge and lasso.
    plot_coefficients(trainx, trainy, 10)


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


'''
2c)
Apply standard linear regression. Return a dictionary with the metrtics and associated values.
'''
def eval_linear1(trainx, trainy, valx, valy, testx, testy):
    model = LinearRegression()
    model.fit(trainx, trainy)

    train_pred = model.predict(trainx)
    val_pred = model.predict(valx)
    test_pred = model.predict(testx)

    return {
        'train-rmse': np.sqrt(mean_squared_error(trainy, train_pred)),
        'train-r2': r2_score(trainy, train_pred),
        'val-rmse': np.sqrt(mean_squared_error(valy, val_pred)),
        'val-r2': r2_score(valy, val_pred),
        'test-rmse': np.sqrt(mean_squared_error(testy, test_pred)),
        'test-r2': r2_score(testy, test_pred)
    }


'''
2d)
Apply Ridge Regression.
'''
def eval_ridge(trainx, trainy, valx, valy, testx, testy, alpha):
    model = Ridge(alpha=alpha)
    model.fit(trainx, trainy)
    train_pred = model.predict(trainx)
    val_pred = model.predict(valx)
    test_pred = model.predict(testx)
    return train_pred, val_pred, test_pred


'''
2e)
Apply Lasso Regression.
'''
def eval_lasso(trainx, trainy, valx, valy, testx, testy, alpha):
    model = Lasso(alpha=alpha)
    model.fit(trainx, trainy)
    train_pred = model.predict(trainx)
    val_pred = model.predict(valx)
    test_pred = model.predict(testx)
    return train_pred, val_pred, test_pred


'''
2f)
Report (using a table) the RMSE and R2 for training, validation, and test for
different λ values.
'''
def report(trainx, trainy, valx, valy, testx, testy, alphas=[0.1, 1, 10, 100, 1000]):
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


'''
2g)
Generate the coefficient path plots (regularization value vs. coefficient value) for both ridge
and lasso models. 
'''
def plot_coefficients(trainx, trainy, alpha):

    _, ax = plt.subplots(1,2, figsize=(10, 5))
    
    # Ridge
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(trainx, trainy)
    ax[0].plot(ridge_model.coef_, 'b')
    ax[0].set_xlabel('Coefficient Index')
    ax[0].set_ylabel('Coefficient Value')
    ax[0].set_title('Ridge Coefficient Path')

    # Lasso
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(trainx, trainy)
    ax[1].plot(lasso_model.coef_, 'r')
    ax[1].set_xlabel('Coefficient Index')
    ax[1].set_ylabel('Coefficient Value')
    ax[1].set_title('Lasso Coefficient Path')

    plt.show()

'''
Execute the main function.
'''
if __name__ == '__main__':
    main()
