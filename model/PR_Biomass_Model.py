from sklearn import linear_model, tree
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def get_data():
    # read in the the biomass data
    df = pd.read_csv('PuertoRico_Biomass_AvgRGB.csv')
    num_rows = len(df)

    data = df[['avg_red', 'avg_green', 'avg_blue', 'biomass']]

    # Populate a numpy array of RGB values with associated biomass
    data_arr = np.zeros((num_rows, 4), dtype=np.float64)
    for i in range(0, num_rows):
        if i % 25000 == 0:
            print(f"{i/num_rows}")
        r = data.loc[i, 'avg_red']
        g = data.loc[i, 'avg_green']
        b = data.loc[i, 'avg_blue']
        bm = data.loc[i, 'biomass']
        data_arr[i] = (r, g, b, bm)

    return data_arr

def split(data, pct_train=.9):
    pct_test = 1 - pct_train

    # shuffle the data around before splitting it
    shuffled_data = data.copy()
    np.random.shuffle(shuffled_data)

    # split into our X and Y variables
    X = shuffled_data[0:len(data), 0:3]
    Y = shuffled_data[0:len(data), 3:4]

    # find the row we split things at
    row_divide = int((len(data) * pct_train) // 1)

    X_train = X[0:row_divide, 0:3]
    Y_train = Y[0:row_divide, 0:1]

    X_test = X[row_divide:len(data), 0:3]
    Y_test = Y[row_divide:len(data), 0:1]

    return X_train, Y_train, X_test, Y_test

def evaluate(Y_test, Y_predictions):
    print(f"Mean squared error: {mean_squared_error(Y_test, Y_predictions)}")
    print(f"Mean absolute error: {mean_absolute_error(Y_test, Y_predictions)}")
    print(f"R^2 score: {r2_score(Y_test, Y_predictions)}")

def main():
    # split data into training and testing section
    print("Creating data set:")
    data = get_data()
    X_train, Y_train, X_test, Y_test = split(data, pct_train=.9)
    print("done")

    # Train a logistic regression model on the training data
    print("Training the model:")
    linreg = linear_model.LinearRegression()
    linreg = linreg.fit(X_train, Y_train)
    print("done")
    
    # evaluate the model for comparison
    print("Evaluating the model:")
    Y_predictions = linreg.predict(X_test)
    evaluate(Y_test, Y_predictions)
    print("done")


# only run the script when run directly
if __name__ == "__main__": 
    main()
