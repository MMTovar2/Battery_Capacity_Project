import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def get_train_and_validation_MSE_and_R2(model):
    # generates a prediction based on our training data to check fit
    Y_pred_train = model.predict(X_train)

    mn_squr_e_train = mean_squared_error(Y_train_scl, Y_pred_train)
    r2_train = r2_score(Y_train_scl, Y_pred_train)

    print('train mean error: ', mn_squr_e_train)
    print('train R2: ', r2_train)

    # generates a prediction based on our validation data to check fit
    Y_pred_valid = model.predict(X_valid)

    mn_squr_e_valid = mean_squared_error(Y_valid_scl, Y_pred_valid)
    r2_valid = r2_score(Y_valid_scl, Y_pred_valid)

    print('valid mean error: ', mn_squr_e_valid)
    print('valid R2: ', r2_valid)

def get_test_MSE_and_R2(model):
    Y_pred_test = model.predict(X_test)

    mn_squr_e_test_a = mean_squared_error(Y_test_scl, Y_pred_test)
    r2_test_a = r2_score(Y_test_scl, Y_pred_test)

    print('test mean error: ', mn_squr_e_test_a)
    print('test R2: ', r2_test_a)
    return Y_pred_test

def training_with_time(model):
    start_time = time.perf_counter()
    model.fit(X_train, Y_train_scl)
    end_time = time.perf_counter()
    print('time: ', end_time - start_time, ' seconds')


data = pd.read_csv('/Users/mtovar/Documents/Matsci 166 (Data Science and Machine Learning)/battery_cycle_level_dataset_CLEAN_FINAL.csv')

size_of_data = len(data)
print('size of dataset: ',size_of_data)

#X data: voltage, temperature, and cycle
X = data[['voltage', 'temperature', 'cycle']]

#Y data: capacity
Y = data['capacity']

#test, train, validation split
X_train, X_test_val, Y_train, Y_test_val = train_test_split(X,Y,test_size=0.3, random_state=14)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_test_val,Y_test_val,test_size=0.5,random_state=13)

#Scalers used
scaler_x = StandardScaler()
scaler_y = StandardScaler()

#pre-processing (mean = 0 + var = 1)
X_train = scaler_x.fit_transform(X_train)

# uses numbers from above without changing them
X_valid = scaler_x.transform(X_valid)
X_test = scaler_x.transform(X_test)

#scale Y to improve fit
Y_train_scl = scaler_y.fit_transform(Y_train.values.reshape(-1,1)).ravel()
Y_valid_scl = scaler_y.transform(Y_valid.values.reshape(-1,1)).ravel()
Y_test_scl = scaler_y.transform(Y_test.values.reshape(-1,1)).ravel()

#Multi-layer perceptron regressor
#used common layer structure for smaller data size (4 layers due to improved performance over 2, 3, and 5 layers),
#used common max iteration, used relu function as it requires less Y pre-processing than sigmoid (current state sigmoid is a poor fit),
# used default learning rate, and increased alpha for a slightly better fit
best_sgd_model = MLPRegressor(random_state=21, hidden_layer_sizes=(64, 32, 16, 8), max_iter=500, activation='relu', learning_rate_init=0.001, alpha=0.001, solver='sgd')
best_lbfgs_model = MLPRegressor(random_state=21, hidden_layer_sizes=(64, 32, 16, 8), max_iter=5000, activation='relu', learning_rate_init=0.001, alpha=0.002, solver='lbfgs')
best_adam_model = MLPRegressor(random_state=21, hidden_layer_sizes=(64, 32, 16, 8), max_iter=500, activation='relu', learning_rate_init=0.001, alpha=0.001)

#linear Model
linear_model = LinearRegression()
print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('SGD')
#time and model train
training_with_time(best_sgd_model)
get_train_and_validation_MSE_and_R2(best_sgd_model)

print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('LBFGS')
#time and model train
training_with_time(best_lbfgs_model)
get_train_and_validation_MSE_and_R2(best_lbfgs_model)

print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('AdaM')
#time and model train
training_with_time(best_adam_model)
get_train_and_validation_MSE_and_R2(best_adam_model)
Y_pred_test = get_test_MSE_and_R2(best_adam_model)

print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

print('Linear')
#time and model train
training_with_time(linear_model)
get_train_and_validation_MSE_and_R2(linear_model)
get_test_MSE_and_R2(linear_model)

print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

#plots loss curve to see if learning rate is too high or low
residuals = Y_test_scl-Y_pred_test

plt.figure(1)
plt.scatter(Y_pred_test, residuals)
size = len(Y_pred_test)
perfect_y = np.zeros(size)
plt.plot(Y_pred_test, perfect_y, color='red', linestyle='--', label='Perfect Fit')

plt.title("Predictions vs Residual")
plt.xlabel('Y predictions (scaled)')
plt.ylabel('Residuals')
plt.legend()

plt.figure(2)
residuals = Y_test_scl-Y_pred_test
plt.hist(residuals, edgecolor='black', bins=20, density=False)
plt.title("Residual vs Number of samples ")
plt.xlabel('Residuals')
plt.ylabel('Count')


plt.figure(3)
plt.plot(best_adam_model.loss_curve_)
plt.title('Loss Curve')
plt.xlabel('Number of Iterations')
plt.ylabel('Magnetude of Error')
plt.show()


