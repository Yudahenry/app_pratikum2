import os
import json
import joblib
import pandas as pd
import streamlit as st

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Prediksi Telco Customer Churn (Machine Learning)",
    page_icon="📊",
    layout="centered"
)

# --- OTOMATIS BUAT FILE CONTOH DATA CSV JIKA BELUM ADA ---
if not os.path.exists("contoh_data_baru.csv"):
    sample_data = pd.DataFrame([
        {
            "tenure": 12, "MonthlyCharges": 65.00, "TotalCharges": 800.00, 
            "SeniorCitizen": 0, "Contract": "Month-to-month", 
            "PaymentMethod": "Electronic check", "InternetService": "DSL", "PaperlessBilling": "Yes"
        },
        {
            "tenure": 34, "MonthlyCharges": 56.95, "TotalCharges": 1889.50, 
            "SeniorCitizen": 0, "Contract": "One year", 
            "PaymentMethod": "Mailed check", "InternetService": "DSL", "PaperlessBilling": "No"
        },
        {
            "tenure": 2, "MonthlyCharges": 70.70, "TotalCharges": 151.65, 
            "SeniorCitizen": 1, "Contract": "Month-to-month", 
            "PaymentMethod": "Electronic check", "InternetService": "Fiber optic", "PaperlessBilling": "Yes"
        },
        {
            "tenure": 45, "MonthlyCharges": 42.30, "TotalCharges": 1840.75, 
            "SeniorCitizen": 0, "Contract": "One year", 
            "PaymentMethod": "Bank transfer (automatic)", "InternetService": "DSL", "PaperlessBilling": "No"
        },
        {
            "tenure": 62, "MonthlyCharges": 89.10, "TotalCharges": 5500.00, 
            "SeniorCitizen": 0, "Contract": "Two year", 
            "PaymentMethod": "Credit card (automatic)", "InternetService": "Fiber optic", "PaperlessBilling": "Yes"
        }
    ])
    sample_data.to_csv("contoh_data_baru.csv", index=False)

# Load Model dan Metadata secara Cached agar efisien
@st.cache_resource
def load_model():
    return joblib.load("model_telco_churn.pkl")

@st.cache_data
def load_info():
    with open("model_info.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
info = load_info()

FITUR_NUMERIK = info["fitur_numerik"]
FITUR_KATEGORI = info["fitur_kategori"]
FITUR = FITUR_NUMERIK + FITUR_KATEGORI

# --- SIDEBAR: INFORMASI MODEL ---
with st.sidebar:
    st.header("📌 Informasi Model")
    st.write(f"**Algoritma:** {info['nama_model']}")
    st.metric("Akurasi Model (Test)", f"{info['metrik']['accuracy']*100:.2f}%")
    st.metric("F1-Score Macro", f"{info['metrik']['f1_macro']:.4f}")
    st.write("---")
    st.write("**Fitur Input Utama:**")
    st.code(", ".join(FITUR))

# --- HEADER UTAMA ---
st.title("📊 Aplikasi Prediksi Telco Customer Churn")
st.write(
    "Aplikasi web interaktif ini menggunakan model *Machine Learning* (Random Forest) "
    "untuk memprediksi apakah pelanggan telekomunikasi akan tetap berlangganan (No) "
    "atau berhenti berlangganan (Yes / Churn)."
)

# Membuat Tab Navigasi
tab1, tab2 = st.tabs(["Prediksi Satu Pelanggan", "Prediksi dari File CSV"])

# --- TAB 1: PREDIKSI SATU PELANGGAN ---
with tab1:
    st.subheader("Formulir Input Data Pelanggan")
    
    with st.form("form_prediksi_ml"):
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.number_input(
                "Lama Berlangganan (Tenure dalam Bulan)", 
                min_value=0, max_value=72, value=12,
                help="Berapa bulan pelanggan telah menggunakan layanan."
            )
            monthly_charges = st.number_input(
                "Biaya Bulanan (Monthly Charges dalam $)", 
                min_value=0.0, max_value=200.0, value=65.0,
                help="Tagihan bulanan yang dibayarkan pelanggan."
            )
            total_charges = st.number_input(
                "Total Akumulasi Biaya (Total Charges dalam $)", 
                min_value=0.0, max_value=10000.0, value=800.0,
                help="Total keseluruhan biaya yang telah dibayarkan."
            )
            senior_citizen = st.selectbox(
                "Status Warga Senior (Senior Citizen)", 
                options=[0, 1], 
                format_func=lambda x: "Ya (Lansia)" if x == 1 else "Tidak"
            )
            
        with col2:
            contract = st.selectbox(
                "Jenis Kontrak (Contract)", 
                options=["Month-to-month", "One year", "Two year"]
            )
            payment_method = st.selectbox(
                "Metode Pembayaran (Payment Method)", 
                options=[
                    "Electronic check", 
                    "Mailed check", 
                    "Bank transfer (automatic)", 
                    "Credit card (automatic)"
                ]
            )
            internet_service = st.selectbox(
                "Layanan Internet (Internet Service)", 
                options=["DSL", "Fiber optic", "No"]
            )
            paperless_billing = st.selectbox(
                "Tagihan Tanpa Kertas (Paperless Billing)", 
                options=["Yes", "No"]
            )
            
        submit_btn = st.form_submit_button("🔍 Jalankan Prediksi Model")

    if submit_btn:
        # Susun input menjadi DataFrame sesuai urutan fitur model
        input_data = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "SeniorCitizen": senior_citizen,
            "Contract": contract,
            "PaymentMethod": payment_method,
            "InternetService": internet_service,
            "PaperlessBilling": paperless_billing
        }])[FITUR]
        
        # Lakukan Prediksi
        prediksi = model.predict(input_data)[0]
        probabilitas = model.predict_proba(input_data)[0]
        kelas_model = model.classes_
        
        prob_dict = dict(zip(kelas_model, probabilitas))
        
        st.markdown("---")
        st.subheader("Hasil Analisis Prediksi:")
        
        if prediksi == "Yes":
            st.error(
                f"⚠️ **Peringatan: Pelanggan Diprediksi CHURN (Berhenti Berlangganan)**\n\n"
                f"Tingkat Keyakinan Model: **{prob_dict['Yes']*100:.2f}%**"
            )
        else:
            st.success(
                f"✅ **Aman: Pelanggan Diprediksi TETAP AKTIF (No Churn)**\n\n"
                f"Tingkat Keyakinan Model: **{prob_dict['No']*100:.2f}%**"
            )
            
        # Tampilkan grafik probabilitas batang
        st.write("Grafik Distribusi Probabilitas:")
        prob_series = pd.Series(prob_dict, name="Probabilitas")
        st.bar_chart(prob_series)

# --- TAB 2: PREDIKSI DARI FILE CSV ---
with tab2:
    st.subheader("Unggah Data Massal (Format CSV)")
    st.write(
        "Unggah file CSV yang berisi kumpulan data pelanggan baru untuk "
        "dianalisis secara otomatis oleh model secara sekaligus."
    )
    
    # 📥 Tombol Unduh Sampel / Dataset Uji Coba (Selalu Tersedia)
    try:
        with open("contoh_data_baru.csv", "rb") as f:
            st.download_button(
                label="📥 Unduh Dataset / Contoh Format File CSV Uji Coba",
                data=f,
                file_name="contoh_data_baru.csv",
                mime="text/csv",
                help="Klik untuk mengunduh sampel data uji coba CSV guna diunggah kembali ke bawah."
            )
    except Exception as e:
        st.warning(f"Gagal memuat file unduhan: {e}")

    st.markdown("---")

    uploaded_file = st.file_uploader("Pilih file CSV pelanggan", type=["csv"])
    
    if uploaded_file is not None:
        df_user = pd.read_csv(uploaded_file)
        st.write("Pratinjau Data yang Diunggah:")
        st.dataframe(df_user.head())
        
        if st.button("🚀 Proses Prediksi Massal"):
            try:
                # Pastikan kolom fitur tersedia
                missing_cols = [col for col in FITUR if col not in df_user.columns]
                if missing_cols:
                    st.error(f"Kolom berikut tidak ditemukan pada CSV Anda: {missing_cols}")
                else:
                    # Jalankan prediksi massal
                    df_user["Prediksi_Churn"] = model.predict(df_user[FITUR])
                    
                    st.success("Prediksi massal berhasil diselesaikan!")
                    st.dataframe(df_user)
                    
                    # Tombol unduh hasil prediksi
                    csv_output = df_user.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Unduh Hasil Prediksi (CSV)",
                        data=csv_output,
                        file_name="hasil_pred_telco_churn.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")