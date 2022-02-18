###################################################
# Source code of Multiple Linear Regression (MLR) #
###################################################

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


# Import dataset
dataset = pd.read_csv('data/Real estate valuation data set.txt', sep='\t')
#print(dataset)


# Split between training set and test set
training_patterns = dataset.head(331)
test_patterns = dataset.tail(83)

x_training = training_patterns[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
y_training = training_patterns['Y']

x_test = test_patterns[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
y_test = test_patterns['Y']


# K-fold cross-validation
cv_prediction_errors = []
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(training_patterns):
    x_training_cv, X_test_cv = training_patterns.iloc[train_index], training_patterns.iloc[test_index]

    x_training_x = x_training_cv.iloc[:, 0:6]
    x_training_y = x_training_cv.iloc[:, 6]

    X_test_x = X_test_cv.iloc[:, 0:6]
    X_test_y = X_test_cv.iloc[:, 6]

    linear_regression = LinearRegression()

    # Training (use k-1 subsets to train)
    linear_regression.fit(x_training_x, x_training_y)

    # Validation (predict over the remaining validation subset)
    y_predicted_MLR = linear_regression.predict(X_test_x)

    # Calculate the prediction percentage error
    errors = abs(y_predicted_MLR - X_test_y)
    cv_prediction_error = 100 * errors.sum() / X_test_y.sum()
    cv_prediction_errors.append(cv_prediction_error)

for cv_prediction_error in cv_prediction_errors:
    print("Fold", cv_prediction_error)
cv_average_prediction_error = sum(cv_prediction_errors)/len(cv_prediction_errors)
print('The average prediction error of cross-validation is E(%)=', cv_average_prediction_error)

linear_regression = LinearRegression()

# Training
linear_regression.fit(x_training, y_training)

# Test
y_predicted_MLR = linear_regression.predict(x_test)

# Calculate the prediction percentage error
errors = abs(y_predicted_MLR - y_test)
mape = 100 * errors.sum() / y_test.sum()
print('The prediction percentage error is E(%)=', mape)

# Scatter plot of the prediction versus real value
plt.scatter(y_test, y_predicted_MLR)
plt.title('Scatter plot of the prediction versus real value for MLR')
plt.xlabel('y_test')
plt.ylabel('y_predicted_MLR')
plt.show()
