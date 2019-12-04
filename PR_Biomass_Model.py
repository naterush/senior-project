from sklearn import linear_model, tree
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def error(y, y_predictions):
    i=0
    arr_len = len(y)
    wrong_count = 0
    while i < arr_len:
        if y[i] != y_predictions[i]:
            wrong_count = wrong_count + 1
        i = i+1
    err = wrong_count / arr_len
    return err

df = pd.read_csv('PuertoRico_Biomass_AvgRGB.csv')
num_rows = len(df)

data = df[['avg_red', 'avg_green', 'avg_blue', 'biomass']]
# distinct_bms = df[['biomass']].drop_duplicates()
# display(distinct_bms)

s = 0
all_nonzero_bms = data[data['biomass'] != 0]['biomass']
for temp_bm in all_nonzero_bms:
    s = s + temp_bm
avg = s / len(all_nonzero_bms)
print("Average Biomass of Areas with Biomass > 0.0: " + str(avg))

# Populate a numpy array of RGB values with associated biomass
data_arr = np.zeros((num_rows, 4), dtype=np.float64)
for i in range(0, num_rows):
    if i % 25000 == 0:
        print("Row " + str(i) + '/' + str(num_rows))
    r = data.loc[i, 'avg_red']
    g = data.loc[i, 'avg_green']
    b = data.loc[i, 'avg_blue']
    bm = data.loc[i, 'biomass']
    data_arr[i] = (r, g, b, bm)

print('data_arr.shape: ', data_arr.shape)

pct_train = 0.9
pct_test = 1 - pct_train

# Shuffle the data to prepare for splitting into training/test sets
shuffled_data = data_arr.copy()
np.random.shuffle(shuffled_data)

X = shuffled_data[0:num_rows, 0:3]
print('X.shape: ', X.shape)
Y = shuffled_data[0:num_rows, 3:4]
print('Y.shape: ', Y.shape)


row_divide = int((num_rows * pct_train) // 1)
print('row_divide: ', row_divide)

X_train = X[0:row_divide, 0:3]
Y_train = Y[0:row_divide, 0:1]
print("X_train")
# display(X_train)
print("Y_train")
# display(Y_train)

X_test = X[row_divide:num_rows, 0:3]
Y_test = Y[row_divide:num_rows, 0:1]

# Train a logistic regression model on the training data
linreg = linear_model.LinearRegression()
linreg = linreg.fit(X_train, Y_train)

Y_predictions = linreg.predict(X_test)
preds = []
pct_errors = []
for i in range(0, len(Y_test)):
    actual_bm = Y_test[i]
    predicted_bm = Y_predictions[i]
    # Difference between predicted biomass and actual biomass FOR ALL POINTS
    diff = abs(actual_bm - predicted_bm)
    # Difference between prediction and actual biomass FOR POINTS WITHOUT BIOMASS
    # ignore_zeros_diff = 0 if (actual_bm == 0.0) else diff
    # Difference between prediction and actual biomass FOR POINTS WITH TRUE BIOMASS
    # ignore_actual_bm_diff = 0 if (actual_bm != 0.0) else diff
    # Prediction error as percentage from true value
    pct_error = 0.0
    if actual_bm != 0.0:
        err = abs(actual_bm-predicted_bm)/actual_bm
        pct_errors.append(err)
    # preds.append([actual_bm, predicted_bm, diff, ignore_zeros_diff, ignore_actual_bm_diff, pct_error])
    preds.append([actual_bm, predicted_bm, diff])

pred_arr = np.array(preds)

avg_error_overall = pred_arr[:, 2].mean()
print("Average error overall (Tonnes/Hectare): " + str(avg_error_overall))

# pct_errors_arr = np.array([n[3] for n in pred_arr if n[1] != 0.0])
pct_errors_arr = np.array(pct_errors)
print(pct_errors_arr)
avg_pct_error = pct_errors_arr.mean()
print("avg_pct_error: ", avg_pct_error)


# avg_error_true_zeros = pred_arr[:, 3].mean()
# print("Average error on areas with 0 measured biomass: " + str(avg_error_true_zeros))

# avg_error_non_zeros = pred_arr[:, 4].mean()
# print("Average error on areas with non-zero biomass: " + str(avg_error_non_zeros))

# Calculate the percentage error among predictions over points WITH TRUE BIOMASS


# Train a decision tree model on the training data
# dt = tree.DecisionTreeClassifier(max_depth=5)
# dt = dt.fit(X_train, Y_train)
