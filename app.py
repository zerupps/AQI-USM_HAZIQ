import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# ---------------------------------
# 1. LOAD AI ASSETS (LSTM)
# ---------------------------------
@st.cache_resource
def load_ml_assets():
    try:
        # Load Model Sniper (5 Minit) & Model Teropong (1 Jam)
        # Sila tukar nama fail .h5 ni mengikut nama yang kau save dalam Jupyter
        model_5min = load_model('top_model_version_5M.keras')
        model_1jam = load_model('top_model_version_1Hour.keras')
        
        # Load scaler yang terbaru
        with open('scaler_multivariate_LSTM_05062026_1Hour.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        return model_5min, model_1jam, scaler
    except Exception as e:
        st.error(f"Gagal load model AI: {e}")
        st.info("Tips: Pastikan fail .h5 dan .pkl ada dalam folder yang sama di Github/Streamlit.")
        st.stop()

model_5min, model_1jam, scaler_lstm = load_ml_assets()

count = st_autorefresh(interval=60000, limit=1000, key="fscounter")

# ---------------------------------
# 2. FUNGSI IPU & STATUS
# ---------------------------------
def calculate_ipu_pm25(pm25_raw):
    """Kira nilai Indeks Pencemar Udara (IPU) berdasarkan nilai PM2.5 mengikut standard USEPA/JAS Malaysia"""
    try:
        pm25 = round(float(pm25_raw), 1)
    except (ValueError, TypeError):
        return 0 

    # Breakpoints rasmi
    if pm25 <= 12.0: 
        return (50 / 12.0) * pm25
    elif pm25 <= 35.4: 
        return ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
    elif pm25 <= 55.4: 
        return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
    elif pm25 <= 150.4: 
        return ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
    elif pm25 <= 250.4: 
        return ((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201
    elif pm25 <= 350.4: 
        return ((400 - 301) / (350.4 - 250.5)) * (pm25 - 250.5) + 301
    elif pm25 <= 500.4: 
        return ((500 - 401) / (500.4 - 350.5)) * (pm25 - 350.5) + 401
    else: 
        return 500

def get_ipu_status(ipu_value):
    """Kembalikan status, warna hex, dan emoji berdasarkan tahap IPU"""
    if ipu_value <= 50: return "Good", "#00b050", "🟢" 
    elif ipu_value <= 100: return "Moderate", "#92d050", "🟡" 
    elif ipu_value <= 200: return "Unhealthy", "#ffff00", "🟠" 
    elif ipu_value <= 300: return "Very Unhealthy", "#ff9900", "🔴" 
    else: return "Hazardous", "#ff0000", "☠️" 

# ---------------------------------
# 3. SAMBUNGAN FIREBASE
# ---------------------------------
if "firebase" in st.secrets:
    firebase_info = dict(st.secrets["firebase"])
else:
    st.error("Secrets 'firebase' tidak dijumpai! Check balik setting kat Streamlit Cloud.")
    st.stop()

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_info)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://aqi-usm-haziq-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

# ---------------------------------
# 4. PAPARAN UI UTAMA
# ---------------------------------
st.title("Air Quality Monitoring Dashboard")
st.write("Location: USM Nibong Tebal")

# Tarik Data Live
live_ref = db.reference('/live').get()

if live_ref:
    last_ts = live_ref.get('timestamp', 'Tiada Data')
    st.caption(f"🕒 Latest Update: {last_ts}")

    current_pm25 = live_ref.get('pm2_5', 0)
    live_ipu = calculate_ipu_pm25(current_pm25)
    status_text, status_color, status_emoji = get_ipu_status(live_ipu)

    st.markdown(
        f"""
        <div style="background-color: {status_color}; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
            <h3 style="margin: 0; color: black;">{status_emoji} Status Air Quality: {status_text} (IPU: {round(live_ipu)})</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", f"{live_ref['temperature']} °C")
    col2.metric("Humidity", f"{live_ref['humidity']} %")
    col3.metric("PM 2.5", f"{live_ref['pm2_5']} µg/m³")
    
# Tarik Data Sejarah (Untuk Graf)
history_ref = db.reference('/history').order_by_key().limit_to_last(500).get()

if history_ref:
    data_list = [val for val in history_ref.values()]
    df = pd.DataFrame(data_list)
    
    with st.sidebar:
        st.divider()
        st.header("📂 Pengurusan Data")
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Backup (CSV)",
            data=csv,
            file_name='data_aqi_usm_backup.csv',
            mime='text/csv',
            help="Simpan data tempat lama untuk analisis tesis nanti."
        )
        
        st.warning("Amaran: Data yang dipadam tidak boleh dikembalikan.")
        if st.button("🗑️ Padam Data Tempat Lama"):
            try:
                db.reference('/history').delete()
                st.success("Firebase dikosongkan!")
                st.rerun()
            except Exception as e:
                st.error(f"Gagal padam: {e}")
    
    st.subheader("Graph Air Quality (5 Minutes Interval)")
    st.line_chart(df.set_index('timestamp')[['pm2_5', 'temperature']])


# ---------------------------------
# 5. PREDICTION AI (LSTM)
# ---------------------------------
st.divider()
st.subheader("🤖 AI Prediction (LSTM Engine)")

# MENU PILIHAN PENGGUNA
pilihan_masa = st.radio(
    "Pilih Tempoh Ramalan Ke Hadapan:",
    ("Next 5 Minutes", "Next 1 hour"),
    horizontal=True
)

mapping_payload = {
    'temperature': 'temp',
    'humidity': 'hum',
    'gas_voltage': 'gas_ppm',
    'pm1_0': 'pm1',
    'pm2_5': 'pm25',
    'pm10_0': 'pm10'
}

fitur_ml = ['temp', 'hum', 'gas_ppm', 'pm1', 'pm25', 'pm10']

if history_ref:
    df = df.rename(columns=mapping_payload)

    if len(df) >= 12:
        try:
            # PENTING: Untuk LSTM, kita MESTI suap 12 sejarah data ke belakang (Look_back = 12)
            data_predict = df.tail(12)[fitur_ml].copy()
            
            # Scaling
            scaled = scaler_lstm.transform(data_predict.values)
            
            # BENTUK 3D UNTUK LSTM: (1 sampel, 12 masa, 6 fitur)
            final_input = scaled.reshape(1, 12, 6)
            
            # Eksekusi AI berdasarkan pilihan
            if pilihan_masa == "Next 5 Minutes":
                prediction = model_5min.predict(final_input, verbose=0)
                raw_pred = prediction.flatten()[0]  # Ambil jawapan tunggal
                label_masa = "In 5 minutes"
            else:
                prediction = model_1jam.predict(final_input, verbose=0)
                raw_pred = prediction.flatten()[-1] # Ambil ramalan terakhir (minit ke-60)
                label_masa = "In 1 Hour"

            # Inverse Transform
            dummy = np.zeros((1, 6))
            dummy[0, 4] = raw_pred
            inversed_data = scaler_lstm.inverse_transform(dummy)
            final_pm25 = inversed_data[0, 4]
            
            # Paparan
            pred_ipu = calculate_ipu_pm25(final_pm25)
            pred_status, pred_color, pred_emoji = get_ipu_status(pred_ipu)
            
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                st.metric(f"PM 2.5 Prediction ({label_masa})", f"{final_pm25:.2f} µg/m³")
                
            with col_pred2:
                st.markdown(
                    f"""
                    <div style="background-color: {pred_color}; padding: 15px; border-radius: 10px; text-align: center;">
                        <h4 style="margin: 0; color: black;">{pred_emoji} Prediction Status</h4>
                        <p style="margin: 0; color: black; font-weight: bold; font-size: 1.1em;">{pred_status} (IPU: {round(pred_ipu)})</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
        except KeyError as e:
            st.error(f"Kolum {e} masih tak jumpa! Sila semak ejaan payload.")
