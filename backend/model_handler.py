from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os

MODEL_PATH = "saved_models/custom_model.h5"

def train_model():
    data = np.loadtxt("uploaded_data/data.csv", delimiter=',', skiprows=1)
    X = data[:, :-1]  # Feature 데이터
    y = data[:, -1]   # Label 데이터 (실제 정답)

    # 데이터 정규화
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    # 테스트 데이터에서 정확도 평가
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"📊 테스트 정확도: {test_accuracy:.4f}")

    model.save(MODEL_PATH)
    return MODEL_PATH

def predict(features):
    model = load_model(MODEL_PATH)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return float(prediction[0][0])
