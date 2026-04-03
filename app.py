import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import joblib

@st.cache_resource
def load_ml_assets():
    try:
        # Cuba guna joblib pula, lagi "ngam" dengan scikit-learn
        model = joblib.load('model_RF_pengkalan.pkl')
        scaler = joblib.load('scaler_multivarite_ipoh.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.info("Tips: Pastikan versi scikit-learn kat laptop sama dengan kat Streamlit Cloud.")
        st.stop()

model_rf, scaler_rf = load_ml_assets()

count = st_autorefresh(interval=60000, limit=1000, key="fscounter")
# 1. Ambil data dari Streamlit Secrets (Format TOML tadi)
# Pastikan header kat Secrets tu adalah [firebase]
if "firebase" in st.secrets:
    firebase_info = dict(st.secrets["firebase"])
else:
    st.error("Secrets 'firebase' tidak dijumpai! Check balik setting kat Streamlit Cloud.")
    st.stop()

# 1. Hubungkan ke Firebase (Hanya sekali)
if not firebase_admin._apps:
    # Kat Streamlit Cloud, kita guna Secrets, tapi untuk test local guna fail JSON
    cred = credentials.Certificate(firebase_info)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://aqi-usm-haziq-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

st.title("Air Quality Monitoring Dashboard")
st.write("Location: USM Nibong Tebal")

# 2. Tarik Data Live
live_ref = db.reference('/live').get()

if live_ref:
    last_ts = live_ref.get('timestamp', 'Tiada Data')
    st.caption(f"🕒 Kemaskini Terakhir: {last_ts}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", f"{live_ref['temperature']} °C")
    col2.metric("Humidity", f"{live_ref['humidity']} %")
    col3.metric("PM 2.5", f"{live_ref['pm2_5']} µg/m³")
    
# 3. Tarik Data Sejarah (Untuk Graf)
history_ref = db.reference('/history').get()
if history_ref:
    # Tukar format Firebase (JSON) ke Pandas DataFrame
    data_list = [val for val in history_ref.values()]
    df = pd.DataFrame(data_list)
    
    st.subheader("Graph Kualiti Udara (15 Minutes Interval)")
    st.line_chart(df.set_index('timestamp')[['pm2_5', 'temperature']])



st.divider()
st.subheader("🤖 AI Prediction (Next 15 Minutes)")

mapping_payload = {
    'temperature': 'temp',
    'humidity': 'hum',
    'gas_voltage': 'gas_v',
    'pm1_0': 'pm1',
    'pm2_5': 'pm25',
    'pm10_0': 'pm10'
}

fitur_ml = ['temp', 'hum', 'gas_v', 'pm1', 'pm25', 'pm10']

if history_ref:
    data_list = [val for val in history_ref.values()]
    df = pd.DataFrame(data_list)
    
    # 3. Rename kolum secara automatik
    df = df.rename(columns=mapping_payload)

    if len(df) >= 8:
        try:
            # 4. Ambil 8 data terakhir
            data_predict = df.tail(8)[fitur_ml].copy()
            
            # 5. PENTING: Tukar gas_v jadi positif (abs) 
            # Sebab raw_v dalam payload kau mungkin negatif
            data_predict['gas_v'] = data_predict['gas_v'] + 0.5
            
            # 6. Scaling & Predict
            scaled = scaler_rf.transform(data_predict.values)
            final_input = scaled.flatten().reshape(1, -1)
            prediction = model_rf.predict(final_input)

            raw_pred = prediction[0]
            dummy = np.zeros((1, 6))
            dummy[0, 4] = raw_pred
            inversed_data = scaler_rf.inverse_transform(dummy)
            final_pm25 = inversed_data[0, 4]
            
            
            st.metric("Ramalan PM 2.5 (15 Min Depan)", f"{final_pm25:.2f} µg/m³")
            
        except KeyError as e:
            st.error(f"Kolum {e} masih tak jumpa! Sila semak ejaan payload.")
