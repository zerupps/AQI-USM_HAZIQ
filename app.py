import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from streamlit_autorefresh import st_autorefresh

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
