import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

MODEL_PATH = "saved_models/custom_model.h5"

def train_model(target_column="Target_A"):
    df = pd.read_csv("uploaded_data/data.csv")

    # 입력 데이터(X)와 정답 데이터(Y) 분리
    X = df.drop(columns=["ID", "Target_A", "Target_B"])
    y = df[target_column]

    # 데이터 정규화
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 데이터 분포 확인
    unique, counts = np.unique(y, return_counts=True)
    print(f"📊 정답(Label) 분포: {dict(zip(unique, counts))}")

    # 데이터 불균형 해결 (SMOTE 적용)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # 모델 정의
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    # 테스트 데이터 평가
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"📊 테스트 정확도: {test_accuracy:.4f}")

    model.save(MODEL_PATH)
    return MODEL_PATH


def predict(features):
    model = load_model(MODEL_PATH)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return float(prediction[0][0])
