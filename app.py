import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd

# 1. Hubungkan ke Firebase (Hanya sekali)
if not firebase_admin._apps:
    # Kat Streamlit Cloud, kita guna Secrets, tapi untuk test local guna fail JSON
    cred = credentials.Certificate("fail-key-firebase-kau.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'URL_DATABASE_FIREBASE_KAU'
    })

st.title("Air Quality Monitoring Dashboard")
st.write("Lokasi: Bilik Haziq, USM Nibong Tebal")

# 2. Tarik Data Live
live_ref = db.reference('/live').get()

if live_ref:
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
    
    st.subheader("Trend Kualiti Udara (15 Minit Sekali)")
    st.line_chart(df.set_index('timestamp')[['pm2_5', 'temperature']])
