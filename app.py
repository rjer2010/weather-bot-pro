import streamlit as st
import pandas as pd
import joblib
import requests
import pytz
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="미녕예보 AI Pro", page_icon="🌤️", layout="wide")

# 스타일링
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🌡️ 미녕예보 AI Pro v3.0")
st.subheader("Stacking Ensemble 기반 정밀 기온 예측 시스템")

# 1. 모델 로드
@st.cache_resource
def load_minyoung_model():
    return joblib.load('minyoung_stack_model.pkl')

try:
    model = load_minyoung_model()
    st.sidebar.success("✅ 미녕 앙상블 모델 로드 완료")
except Exception as e:
    st.sidebar.error(f"❌ 모델 로드 실패: {e}")

# 2. 날씨 데이터 수집 및 예측 함수
def fetch_and_predict():
    API_KEY = st.secrets["WEATHER_KEY"] # Streamlit Cloud의 Secrets에 넣어야 합니다.
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    
    seoul_tz = pytz.timezone('Asia/Seoul')
    base_date = datetime.now(seoul_tz).strftime("%Y%m%d")
    
    params = {
        'serviceKey': API_KEY,
        'dataType': 'JSON', 'base_date': base_date, 'base_time': '0500', 
        'nx': '55', 'ny': '127', 'numOfRows': 500 
    }
    
    res = requests.get(url, params=params).json()
    items = res['response']['body']['items']['item']
    
    data = {}
    for item in items:
        t = item['fcstTime']
        if t not in data: data[t] = {}
        if item['category'] == 'TMP': data[t]['TMP'] = float(item['fcstValue'])
        if item['category'] == 'REH': data[t]['REH'] = float(item['fcstValue'])
        if item['category'] == 'WSD': data[t]['WSD'] = float(item['fcstValue'])

    # 데이터프레임 변환 및 예측
    df = pd.DataFrame(data).T.dropna()
    df.columns = ['TMP', 'REH', 'WSD'] # 모델 학습 시 컬럼명과 맞춰야 함
    
    # 미녕 모델로 진짜 예측 실행! (수식이 아니라 모델이 직접 계산)
    df['PRED'] = model.predict(df[['TMP', 'REH', 'WSD']]) 
    return df

# 3. 메인 화면 구성
if st.button('실시간 예측 업데이트'):
    with st.spinner('미녕 AI가 복합 모델을 분석 중입니다...'):
        forecast_df = fetch_and_predict()
        
        # 최신 예보 메트릭
        current_pred = forecast_df['PRED'].iloc[0]
        st.metric(label="현재 시점 정밀 예측 기온", value=f"{current_pred:.2f} °C", delta=f"{current_pred - forecast_df['TMP'].iloc[0]:.2f} (AI 보정치)")
        
        # 그래프 표시
        st.write("### 📈 시간대별 기온 변화 (AI 예측)")
        st.line_chart(forecast_df[['TMP', 'PRED']])
        
        # 상세 데이터 표
        with st.expander("상세 분석 데이터 보기"):
            st.dataframe(forecast_df.style.highlight_max(axis=0))

else:
    st.info("위 버튼을 눌러 실시간 앙상블 분석을 시작하세요.")
