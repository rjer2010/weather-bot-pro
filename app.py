import streamlit as st
import pandas as pd
import joblib
import requests
import pytz
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="미녕예보 AI Pro", page_icon="🌤️", layout="wide")

# 스타일링
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🌡️ 미녕예보 AI Pro v3.2")
st.subheader("Stacking Ensemble 기반 정밀 기온 예측 시스템")

# 2. 모델 로드
@st.cache_resource
def load_minyoung_model():
    return joblib.load('minyoung_stack_model.pkl')

try:
    model = load_minyoung_model()
    st.sidebar.success("✅ 미녕 앙상블 모델 로드 완료")
except Exception as e:
    st.sidebar.error(f"❌ 모델 로드 실패: {e}")

# 3. 날씨 데이터 수집 및 예측 함수
def fetch_and_predict():
    if "WEATHER_KEY" not in st.secrets:
        st.error("❌ Secrets에 'WEATHER_KEY'가 설정되지 않았습니다.")
        return pd.DataFrame()

    API_KEY = st.secrets["WEATHER_KEY"]
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    
    seoul_tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(seoul_tz)
    base_date = now.strftime("%Y%m%d")
    
    params = {
        'serviceKey': API_KEY,
        'dataType': 'JSON', 
        'base_date': base_date, 
        'base_time': '0500', 
        'nx': '55', 'ny': '127', 
        'numOfRows': 500 
    }
    
    try:
        response = requests.get(url, params=params)
        res = response.json()
        
        # 기상청 응답 구조 체크
        if 'response' in res and 'body' in res['response'] and 'items' in res['response']['body']:
            items = res['response']['body']['items']['item']
        else:
            header = res.get('response', {}).get('header', {})
            error_msg = header.get('resultMsg', '데이터를 찾을 수 없음')
            error_code = header.get('resultCode', 'Unknown')
            
            st.warning(f"⚠️ 기상청 API 응답 확인: {error_msg}")
            if error_code == '03':
                st.info("💡 API 키가 아직 승인되지 않았습니다. (최대 2시간 소요)")
            elif error_code == '30':
                st.info("💡 서비스 키 등록 오류입니다. Secrets의 키를 확인해 주세요.")
            return pd.DataFrame()

        data = {}
        for item in items:
            t = item['fcstTime']
            if t not in data: data[t] = {}
            if item['category'] == 'TMP': data[t]['TMP'] = float(item['fcstValue'])
            if item['category'] == 'REH': data[t]['REH'] = float(item['fcstValue'])
            if item['category'] == 'WSD': data[t]['WSD'] = float(item['fcstValue'])

        df = pd.DataFrame(data).T.dropna()
        if not df.empty:
            df.columns = ['TMP', 'REH', 'WSD']
            # 미녕님의 진짜 스태킹 모델 예측 수행
            df['PRED'] = model.predict(df[['TMP', 'REH', 'WSD']])
            return df
        return pd.DataFrame()
            
    except Exception as e:
        st.error(f"❌ 데이터 처리 중 오류: {e}")
        return pd.DataFrame()

# 4. 메인 화면 구성
if st.button('🚀 실시간 앙상블 예측 시작'):
    with st.spinner('미녕 AI 분석 중...'):
        forecast_df = fetch_and_predict()
        
        if not forecast_df.empty:
            current_pred = forecast_df['PRED'].iloc[0]
            current_raw = forecast_df['TMP'].iloc[0]
            
            col1, col2 = st.columns(2)
            col1.metric("AI 예측 기온", f"{current_pred:.2f} °C", f"{current_pred-current_raw:.2f} AI보정")
            col2.metric("기상청 예보", f"{current_raw:.1f} °C")
            
            st.write("### 📈 기온 변화 그래프")
            st.line_chart(forecast_df[['TMP', 'PRED']])
        else:
            st.info("데이터를 가져오는 중입니다. 잠시 후 다시 시도하세요.")
else:
    st.info("버튼을 눌러 분석을 시작하세요.")
