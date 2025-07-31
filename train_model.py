import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
import joblib

DATA_PATH = 'data\energy.csv'

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. please place ypur energy consumption data(*.csv) there")


df = pd.read_csv(DATA_PATH)

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

X_train , X_test, y_train, y_test = train_test_split(X, y , test_size=0.2,random_state=42)

# Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_Scaled = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Dense(64, activation = 'relu', input_shape = (X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss = 'mean_squared_error', metrics=['mae'])
model.fit(X_train,y_train,epochs=50, batch_size=32, validation_split=0.1)

loss , mae = model.evaluate(X_test,y_test)

print(f"Test MAE:{mae}")
model.save("energy_model.h5")
joblib.dump(scaler, 'scaler.save')
print("Model and scaler saved")