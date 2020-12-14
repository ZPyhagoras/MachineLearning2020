import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

dataset_path = 'C:\\Users\\DWQE\\LinuxFiles\\MachineLearning2020\\PersonalProject\\Dataset\\'

train_data = pd.read_csv(dataset_path + 'processed_train.csv')
test_data = pd.read_csv(dataset_path + 'processed_test.csv')

X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :]

mm = MinMaxScaler()
mmX_train = mm.fit_transform(X_train)
mmX_test = mm.fit_transform(X_test)

'''
lda = LinearDiscriminantAnalysis(n_components=15)
lda.fit(mmX, y.astype('int'))
ldaX = lda.transform(mmX)
'''

'''
test_size1 = 0.2
seed = 8
X_train, X_test, y_train, y_test = train_test_split(mmX, y, test_size=test_size1, random_state=seed)
'''

knn = MLPRegressor(activation='logistic')
knn.fit(mmX_train, y_train)
knn_pred = knn.predict(mmX_test)
e_knn = np.exp(knn_pred)
test_data['knn_predict'] = e_knn

rf = RandomForestRegressor(criterion='mae')
rf.fit(mmX_train, y_train)
rf_pred = rf.predict(mmX_test)
e_rf = np.exp(rf_pred)
test_data['random-forest_predict'] = e_rf


linear = LinearRegression()
linear.fit(mmX_train, y_train)
linear_pred = linear.predict(mmX_test)
e_linear = np.exp(linear_pred)
test_data['linear_predict'] = e_linear

test_data.to_csv(dataset_path + 'predicted.csv')
