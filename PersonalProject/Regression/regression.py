import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

dataset_path = 'C:\\Users\\DWQE\\LinuxFiles\\MachineLearning2020\\PersonalProject\\Dataset\\'

data = pd.read_csv(dataset_path + 'Processedtrain.csv')
X, y = data.iloc[:, :-2].values, data.iloc[:, -1].values
test_size1 = 0.2
seed = 8
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size1, random_state=seed)

e_y_test = np.exp(y_test)
ts = len(list(e_y_test))

knn = MLPRegressor(activation='logistic')
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
e_knn = np.exp(knn_pred)
print('Regression ->', 'R^2:', r2_score(y_test, knn_pred),
      'MAE:', sum(list(np.maximum(e_knn - e_y_test, e_y_test - e_knn))) / ts)

rf = RandomForestRegressor(criterion='mae')
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
e_rf = np.exp(rf_pred)
print('RandomForest ->', 'R^2:', r2_score(y_test, rf_pred),
      'MAE:', sum(list(np.maximum(e_rf - e_y_test, e_y_test - e_rf))) / ts)

linear = LinearRegression()
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)
e_linear = np.exp(linear_pred)
print('Linear ->', 'R^2:', r2_score(y_test, linear_pred),
      'MAE:', sum(list(np.maximum(e_linear - e_y_test, e_y_test - e_linear))) / ts)
