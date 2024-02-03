# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load your training dataset
train_file_path = 'C:/Users/mohan/Downloads/archive/train.csv'
train_data = pd.read_csv(train_file_path)

# Load your testing dataset
test_file_path = 'C:/Users/mohan/Downloads/archive/test.csv'
test_data = pd.read_csv(test_file_path)

# Split the training data into features (X_train) and target variable (y_train)
X_train = train_data[['TotalBsmtSF', 'BedroomAbvGr', 'BsmtFullBath']]
y_train = train_data['SalePrice']

# Impute missing values in training data
imputer_train = SimpleImputer(strategy='mean')
X_train_imputed = imputer_train.fit_transform(X_train)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the imputed training data
model.fit(X_train_imputed, y_train)

# Split the testing data into features (X_test) - without 'SalePrice'
X_test = test_data[['TotalBsmtSF', 'BedroomAbvGr', 'BsmtFullBath']]

# Impute missing values in testing data
imputer_test = SimpleImputer(strategy='mean')
X_test_imputed = imputer_test.fit_transform(X_test)

# Make predictions on the imputed test set
y_pred = model.predict(X_test_imputed)

# Note: Since you don't have 'SalePrice' in the test_data, you can't compare predictions directly.

# Visualize the predictions
# Visualize only the predicted values with linear regression line
plt.scatter(y_pred, X_test['TotalBsmtSF'], color='blue', label='Predicted SalePrice')
plt.xlabel('Predicted SalePrice')
plt.ylabel('TotalBsmtSF')
plt.legend()
plt.show()
