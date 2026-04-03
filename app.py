import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from streamlit_autorefresh import st_autorefresh

@st.cache_resource
def load_ml_assets():
    # Pastikan nama fail ni sama dengan apa yang kau save
    with open('model_RF_pengkalan.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_multivarite_ipoh.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

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

# Pastikan data history ada sekurang-kurangnya 8 baris
if len(df) >= 8:
    # 1. Ambil 8 data terakhir
    # Susun ikut urutan yang betul (Contoh: Temp, Hum, PM2.5)
    last_8_data = df.tail(8)[['temperature', 'humidity', 'pm2_5']].values
    
    # 2. Scaling (Wajib sebab model belajar guna data scaled)
    last_8_scaled = scaler_rf.transform(last_8_data)
    
    # 3. Flatten data jadi 1D array (Sebab RF lookback 8 selalunya guna input rata)
    # Contoh: 8 rows x 3 features = 24 input features
    input_features = last_8_scaled.flatten().reshape(1, -1)
    
    # 4. Predict!
    prediction = model_rf.predict(input_features)
    
    # 5. Tampilkan Hasil
    # Tukar balik hasil prediction ke unit asal kalau kau scale target masa training
    pred_val = prediction[0] 
    
    col_pred, col_status = st.columns(2)
    col_pred.metric("Ramalan PM 2.5 Seterusnya", f"{pred_val:.2f} µg/m³")
    
    # Status Alert
    if pred_val > 50:
        st.warning("⚠️ Amaran: Kualiti udara dijangka merosot!")
    else:
        st.success("✅ Kualiti udara dijangka kekal bersih.")
else:
    st.info("Sedang mengumpul data yang cukup (minima 8 data) untuk membuat ramalan...")
