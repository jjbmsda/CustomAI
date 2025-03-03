import streamlit as st
import requests
import json

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
    data = {"features": list(map(float, features))}  # 숫자 리스트로 변환
    response = requests.post(f"{API_URL}/predict/", json=data)
    
    if response.status_code == 200:
        result = response.json()
        if "prediction" in result:
            st.write("예측 결과:", result["prediction"])
        else:
            st.error(f"예측 오류: {result}")
    else:
        st.error(f"서버 오류: {response.status_code}")
