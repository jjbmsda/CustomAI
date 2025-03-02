import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "custom_model.h5")

# 간단한 MLP 모델 생성
def create_model(input_shape=10, layers=[(64, 'relu'), (32, 'relu')], output_units=1):
    model = Sequential()
    model.add(Dense(layers[0][0], activation=layers[0][1], input_shape=(input_shape,)))
    
    for units, activation in layers[1:]:
        model.add(Dense(units, activation=activation))
    
    model.add(Dense(output_units, activation="sigmoid"))
    return model

# 가상의 데이터로 학습
def train_model():
    X_train = np.random.rand(100, 10)  # 100개의 샘플, 10개의 특징
    y_train = np.random.randint(0, 2, 100)  # 이진 분류
    
    model = create_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10, batch_size=16)
    
    model.save(MODEL_PATH)
    return MODEL_PATH

# 예측 수행
def predict(features):
    model = tf.keras.models.load_model(MODEL_PATH)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return float(prediction[0][0])
