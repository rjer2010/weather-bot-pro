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

st.title("🌡️ 미녕예보 AI Pro v3.1")
st.subheader("Stacking Ensemble 기반 정밀 기온 예측 시스템")

# 1. 모델 로드
@st.cache_resource
def load_minyoung_model():
    # 업로드하신 pkl 파일 이름과 정확히 일치해야 합니다.
    return joblib.load('minyoung_stack_model.pkl')

try:
    model = load_minyoung_model()
    st.sidebar.success("✅ 미녕 앙상블 모델 로드 완료")
except Exception as e:
    st.sidebar.error(f"❌ 모델 로드 실패: {e}")

# 2. 날씨 데이터 수집 및 예측 함수
def fetch_and_predict():
    if "WEATHER_KEY" not in st.secrets:
        st.error("❌ Secrets에 'WEATHER_KEY'가 설정되지 않았습니다.")
        return pd.DataFrame()

    API_KEY = st.secrets["WEATHER_KEY"]
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    
    seoul_tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(seoul_tz)
    
    # 기상청 단기예보는 0500시 데이터가 가장 안정적입니다.
    base_date = now.strftime("%Y%m%d")
    base_time = "0500"
    
    params = {
        'serviceKey': API_KEY,
        'dataType': 'JSON', 
        'base_date': base_date, 
        'base_time': base_time, 
        'nx': '55', 'ny': '127', 
        'numOfRows': 500 
    }
    
    try:
        response = requests.get(url, params=params)
        res = response.json()
        
        # [중요] 기상청 응답 구조 확인 (KeyError: 'item' 방어)
        if 'response' in res and 'body' in res['response'] and 'items' in res['response']['body']:
            items = res['response']['body']['items']['item']
        else:
            # 에러 원인 분석 및 출력
            header = res.get('response', {}).get('header', {})
            error_msg = header.get('resultMsg', '데이터를 찾을 수 없음')
            error_code = header.get('resultCode', 'Unknown')
            
            st.warning(f"⚠️ 기상청 API 응답 확인: {error_msg} (코드: {error_code})")
            if error_code == '03':
                st.info("💡 공공데이터포털에서 API 키가 아직 승인/활성화되지 않았을 수 있습니다. (최대 2시간 소요)")
            elif error_code == '30':
                st.info("💡 서비스 키 등록 오류입니다. Secrets의 키가 올바른지 다시
