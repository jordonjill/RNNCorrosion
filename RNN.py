import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN
import tensorflow as tf
import matplotlib.pyplot as plt
import random
random.seed(42)

#%% Read and preprocess data
data = pd.read_excel("Corrosion Data.xlsx")
# Select features (2nd, 3rd, 4th columns) and target (last column)
features = data.iloc[:, 1:4].values
target = data.iloc[:, -1].values
sc = StandardScaler()
features_scaled = sc.fit_transform(features)

#%% Create sequences for LSTM
def create_sequences(features, target, time_steps=1):
    X, y = [], []
    for i in range(len(features) - time_steps+1):
        seq_x = features[i:(i + time_steps)]
        seq_y = target[i + time_steps-1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

time_steps = 3 #lookback period, it means lookback is 2.
X, y = create_sequences(features_scaled, target, time_steps)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

#%% Build and train LSTM model
np.random.seed(2019)
model = Sequential()
model.add(LSTM(units=3, activation='sigmoid', input_shape=(time_steps, features.shape[1])))
#For SimpleRNN, just replace LSTM with SimpleRNN 
model.add(Dense(units=1))
optimizer =tf.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')
tf.random.set_seed(42)
history=model.fit(X_train, y_train, epochs=100, batch_size=2)
predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

#%% Evaluate LSTM model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
print("Model\t\t RMSE \t\t MSE \t\t MAPE \t\t R2")
print("""LSTM-Test \t {:.2f} \t\t {:.2f} \t\t{:.2f}% \t\t{:.2f}""".format(
    np.sqrt(mean_squared_error(y_test, predictions_test)), mean_squared_error(y_test, predictions_test),
    mean_absolute_percentage_error(y_test, predictions_test) * 100, r2_score(y_test, predictions_test)))
print("\n")
print("Model\t\t RMSE \t\t MSE \t\t MAPE \t\t R2")
print("""LSTM-Train \t {:.2f} \t\t {:.2f} \t\t{:.2f}% \t\t{:.2f}""".format(
    np.sqrt(mean_squared_error(y_train, predictions_train)), mean_squared_error(y_train, predictions_train),
    mean_absolute_percentage_error(y_train, predictions_train) * 100, r2_score(y_train, predictions_train)))
print("\n")

#%% Plot the results
plt.figure(figsize=(10, 6))
plt.plot(data['Time (days)'][time_steps-1:time_steps+train_size-1], predictions_train, marker='*', label='Train Predictions')
plt.plot(data['Time (days)'][time_steps+train_size-1:], predictions_test, marker='+', label='Test Predictions')
plt.plot(data['Time (days)'], target, label='Actual Values', marker='o')
plt.legend()
plt.show()