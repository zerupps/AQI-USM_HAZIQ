import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import joblib
import numpy as np

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

# --- API ---
def calculate_ipu_pm25(pm25_raw):
    """Kira nilai Indeks Pencemar Udara (IPU) berdasarkan nilai PM2.5"""
    try:
        # Bundarkan ke 1 titik perpuluhan ikut standard rasmi
        pm25 = round(float(pm25_raw), 1)
    except (ValueError, TypeError):
        return 0 

    # Pengiraan mengikut jadual breakpoint Malaysia yang tepat
    if pm25 <= 12.0:
        # Kategori: Baik (0-50)
        return (50 / 12.0) * pm25
    
    elif pm25 <= 35.0:
        # Kategori: Sederhana (51-100)
        return ((100 - 51) / (35.0 - 12.1)) * (pm25 - 12.1) + 51
    
    elif pm25 <= 55.0:
        # Kategori: Tidak Sihat (101-200)
        return ((200 - 101) / (55.0 - 35.1)) * (pm25 - 35.1) + 101
    
    elif pm25 <= 150.0:
        # Kategori: Sangat Tidak Sihat (201-300)
        return ((300 - 201) / (150.0 - 55.1)) * (pm25 - 55.1) + 201
    
    elif pm25 <= 250.0:
        # Kategori: Berbahaya (301-500)
        return ((500 - 301) / (250.0 - 150.1)) * (pm25 - 150.1) + 301
    
    else:
        # Kecemasan
        return 500

def get_ipu_status(ipu_value):
    """Kembalikan status, warna hex, dan emoji berdasarkan tahap IPU"""
    if ipu_value <= 50:
        return "Good", "#00b050", "🟢" # Hijau
    elif ipu_value <= 100:
        return "Moderate", "#92d050", "🟡" # Kuning Kehijauan
    elif ipu_value <= 200:
        return "Unhealthy", "#ffff00", "🟠" # Kuning
    elif ipu_value <= 300:
        return "Very Unhealthy", "#ff9900", "🔴" # Jingga
    else:
        return "Hazardous", "#ff0000", "☠️" # Merah
# ---------------------------------
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
    
# 3. Tarik Data Sejarah (Untuk Graf)
history_ref = db.reference('/history').get()
if history_ref:
    # Tukar format Firebase (JSON) ke Pandas DataFrame
    data_list = [val for val in history_ref.values()]
    df = pd.DataFrame(data_list)
    # --- TAMBAHAN: PENGURUSAN DATA DI SIDEBAR ---
    with st.sidebar:
        st.divider()
        st.header("📂 Pengurusan Data")
        
        # 1. Butang Download (Backup data tempat lama sebelum delete)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Backup (CSV)",
            data=csv,
            file_name='data_aqi_usm_backup.csv',
            mime='text/csv',
            help="Simpan data tempat lama untuk analisis tesis nanti."
        )
        
        # 2. Butang Clear Data
        st.warning("Amaran: Data yang dipadam tidak boleh dikembalikan.")
        if st.button("🗑️ Padam Data Tempat Lama"):
            try:
                # Padam node /history di Firebase
                db.reference('/history').delete()
                st.success("Firebase dikosongkan!")
                # Refresh dashboard supaya graf bermula baru
                st.rerun()
            except Exception as e:
                st.error(f"Gagal padam: {e}")
    # --------------------------------------------
    
    st.subheader("Graph Air Quality (5 Minutes Interval)")
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

    if len(df) >= 12:
        try:
            # 4. Ambil 8 data terakhir
            data_predict = df.tail(12)[fitur_ml].copy()
            
            # 5. PENTING: Tukar gas_v jadi positif (abs) 
            # Sebab raw_v dalam payload kau mungkin negatif
            #data_predict['gas_v'] = data_predict['gas_v'] + 0.5
            
            # 6. Scaling & Predict
            scaled = scaler_rf.transform(data_predict.values)
            final_input = scaled.flatten().reshape(1, -1)
            prediction = model_rf.predict(final_input)

            raw_pred = prediction[0]
            dummy = np.zeros((1, 6))
            dummy[0, 4] = raw_pred
            inversed_data = scaler_rf.inverse_transform(dummy)
            final_pm25 = inversed_data[0, 4]

        # ... kod atas ...
            raw_pred = prediction[0]
            dummy = np.zeros((1, 6))
            dummy[0, 4] = raw_pred
            inversed_data = scaler_rf.inverse_transform(dummy)
            final_pm25 = inversed_data[0, 4]
            
            # --- MULA LETAK KOD BARU KAT SINI ---
            pred_ipu = calculate_ipu_pm25(final_pm25)
            pred_status, pred_color, pred_emoji = get_ipu_status(pred_ipu)
            
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                st.metric("PM 2.5 Prediction (In 15 minutes)", f"{final_pm25:.2f} µg/m³")
                
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
            # --- TAMAT KOD BARU ---
            
        except KeyError as e:
            st.error(f"Kolum {e} masih tak jumpa! Sila semak ejaan payload.")
            
            #st.metric("Ramalan PM 2.5 (15 Min Depan)", f"{final_pm25:.2f} µg/m³")
            
        except KeyError as e:
            st.error(f"Kolum {e} masih tak jumpa! Sila semak ejaan payload.")
