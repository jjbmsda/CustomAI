import streamlit as st
import requests

# FastAPI 서버 주소
API_URL = "http://localhost:8000"

st.title("맞춤형 AI 모델 서비스")

# 데이터 업로드
st.subheader("데이터 업로드")
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{API_URL}/upload-data/", files=files)
    st.write(response.json())

# 모델 학습
if st.button("모델 학습 시작"):
    response = requests.post(f"{API_URL}/train-model/")
    st.write(response.json())

# 예측 수행
st.subheader("모델 예측")
features = [st.number_input(f"Feature {i+1}", value=0.0) for i in range(10)]

if st.button("예측 실행"):
    response = requests.post(f"{API_URL}/predict/", json={"features": features})
    st.write("예측 결과:", response.json()["prediction"])
