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
    st.write("업로드된 데이터 미리보기:", df.head())

    # 첫 번째 행의 Feature 값 가져오기
    input_features = df.iloc[0, :-1].tolist()  # 마지막 열(Label) 제외
    actual_label = df.iloc[0, -1]  # 마지막 열(Label) 가져오기

    st.subheader("자동 예측 데이터")
    st.write("예측에 사용될 데이터:", input_features)
    st.write("실제 정답 (Label):", actual_label)  # 실제 정답 표시

    if st.button("예측 실행"):
        data = {"features": input_features}
        response = requests.post(f"{API_URL}/predict/", json=data)

        if response.status_code == 200:
            result = response.json()
            if "prediction" in result:
                predicted_value = 1 if result["prediction"] > 0.5 else 0  # 확률을 0 또는 1로 변환
                st.write("예측 결과:", result["prediction"])
                st.write(f"📌 모델 예측값: {predicted_value}, 실제 정답: {actual_label}")

                # 예측이 맞았는지 확인
                if predicted_value == actual_label:
                    st.success("✅ 예측이 정확합니다!")
                else:
                    st.error("❌ 예측이 틀렸습니다.")
            else:
                st.error(f"예측 오류: {result}")
        else:
            st.error(f"서버 오류: {response.status_code}")

