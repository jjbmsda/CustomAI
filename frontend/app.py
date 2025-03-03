import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

st.title("맞춤형 AI 모델 서비스")

# 데이터 업로드
st.subheader("CSV 파일 업로드")
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("업로드된 데이터 미리보기:", df.head())  # 파일 내용 확인

    # 첫 번째 행을 예측 데이터로 사용
    input_features = df.iloc[0, :-1].tolist()  # 마지막 열(label) 제외

    st.subheader("자동 예측 데이터")
    st.write("예측에 사용될 데이터:", input_features)

    # 예측 실행
    if st.button("예측 실행"):
        data = {"features": input_features}
        response = requests.post(f"{API_URL}/predict/", json=data)

        if response.status_code == 200:
            result = response.json()
            if "prediction" in result:
                st.write("예측 결과:", result["prediction"])
            else:
                st.error(f"예측 오류: {result}")
        else:
            st.error(f"서버 오류: {response.status_code}")
