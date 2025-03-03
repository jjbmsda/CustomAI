import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

st.title("맞춤형 AI 모델 서비스")

# CSV 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("업로드된 데이터 미리보기:", df.head())

    # 사용자에게 예측할 행 선택 옵션 제공
    row_index = st.number_input("예측할 행 번호를 선택하세요 (0부터 시작)", min_value=0, max_value=len(df)-1, step=1, value=0)

    # 선택된 행의 Feature 값 가져오기
    input_features = df.iloc[row_index, :-1].tolist()  # 마지막 열(Label) 제외
    actual_label = df.iloc[row_index, -1]  # 실제 정답

    st.write(f"🔹 선택한 행의 데이터 (Index {row_index}):", input_features)
    st.write(f"✅ 실제 정답 (Label):", actual_label)

    # 예측 실행
    if st.button("예측 실행"):
        data = {"features": input_features}
        response = requests.post(f"{API_URL}/predict/", json=data)

        if response.status_code == 200:
            result = response.json()
            predicted_value = 1 if result["prediction"] > 0.5 else 0  # 확률을 0 또는 1로 변환
            st.write("예측 결과:", result["prediction"])
            st.write(f"📌 모델 예측값: {predicted_value}, 실제 정답: {actual_label}")

            # 예측이 맞았는지 확인
            if predicted_value == actual_label:
                st.success("✅ 예측이 정확합니다!")
            else:
                st.error("❌ 예측이 틀렸습니다.")
        else:
            st.error(f"서버 오류: {response.status_code}")
