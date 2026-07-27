# MENGIMPOR PUSTAKA (LIBRARY) UTAMA
import streamlit as st # Untuk antarmuka aplikasi web
import pandas as pd # Untuk memproses data tabular
import numpy as np # Untuk komputasi array matematis
from sklearn.ensemble import RandomForestClassifier # Algoritma utama
from sklearn.preprocessing import LabelEncoder, StandardScaler # Pra-pemrosesan
from sklearn.model_selection import train_test_split # Membagi data
from imblearn.over_sampling import SMOTE # Menangani imbalanced data
import warnings
warnings.filterwarnings('ignore')

# konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Risiko Stroke",
    page_icon="🧠",
    layout="centered"
)

# judul aplikasi
st.title("🧠 Klasifikasi Penyakit Risiko Stroke")
st.markdown("**Menggunakan Algoritma Random Forest**")
st.markdown("---")

# fungsi untuk melatih model
@st.cache_resource
def load_and_train():
    # baca dataset
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")

    # MENGHAPUS FITUR YANG TIDAK RELEVAN
    # Kolom 'id' dihapus karena tidak memiliki nilai medis terhadap risiko stroke
    df.drop(columns=['id'], inplace=True)

    # Menghapus data gender 'Other' karena jumlahnya hanya 1 baris
    df = df[df['gender'] != 'Other']

    # MENANGANI NILAI KOSONG (MISSING VALUES)
    # Nilai kosong pada BMI diisi dengan Median (nilai tengah) agar kebal terhadap pencilan (outlier)
    bmi_median = df['bmi'].median()
    df['bmi'].fillna(bmi_median, inplace=True)

    # pastikan tidak ada nilai kosong yang tersisa sebelum SMOTE
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['avg_glucose_level'] = pd.to_numeric(df['avg_glucose_level'], errors='coerce')
    df = df.fillna(df.median(numeric_only=True))
    df = df.dropna()

    # LABEL ENCODING PADA FITUR KATEGORIKAL
    # Mengubah data teks (seperti 'Male'/'Female') menjadi angka biner agar bisa diproses mesin
    le_gender    = LabelEncoder()
    le_married   = LabelEncoder()
    le_work      = LabelEncoder()
    le_residence = LabelEncoder()
    le_smoking   = LabelEncoder()

    df['gender']         = le_gender.fit_transform(df['gender'])
    df['ever_married']   = le_married.fit_transform(df['ever_married'])
    df['work_type']      = le_work.fit_transform(df['work_type'])
    df['Residence_type'] = le_residence.fit_transform(df['Residence_type'])
    df['smoking_status'] = le_smoking.fit_transform(df['smoking_status'])

    # pisahkan fitur dan label
    X = df.drop(columns=['stroke'])
    y = df['stroke']

    # PENANGANAN IMBALANCED DATA DENGAN SMOTE
    # Menciptakan data sintetis pada kelas minoritas agar jumlah data seimbang
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # PEMBAGIAN DATA LATIH DAN DATA UJI
    # Membagi 90% Data Latih dan 10% Data Uji dengan proporsi kelas yang setara (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.10, random_state=42, stratify=y_res
    )

    # NORMALISASI SKALA FITUR (SCALING)
    # Menyamakan rentang nilai antar fitur agar algoritma tidak bias pada angka besar
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    # MELATIH ALGORITMA RANDOM FOREST
    # Menggunakan parameter terbaik hasil hyperparameter tuning (GridSearchCV)
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train_sc, y_train)

    return model, scaler, le_gender, le_married, le_work, le_residence, le_smoking

# muat model
with st.spinner("memuat model, harap tunggu..."):
    model, scaler, le_gender, le_married, le_work, le_residence, le_smoking = load_and_train()

st.success("model berhasil dimuat")
st.markdown("---")

# form input data pasien
st.subheader("📋 Masukkan Data Pasien")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Jenis Kelamin",
        options=["Female", "Male"]
    )
    age = st.number_input(
        "Usia (tahun)",
        min_value=1, max_value=120, value=45, step=1
    )
    hypertension = st.selectbox(
        "Riwayat Hipertensi",
        options=["Tidak", "Ya"]
    )
    heart_disease = st.selectbox(
        "Riwayat Penyakit Jantung",
        options=["Tidak", "Ya"]
    )
    ever_married = st.selectbox(
        "Status Pernikahan",
        options=["No", "Yes"]
    )

with col2:
    work_type = st.selectbox(
        "Jenis Pekerjaan",
        options=["Govt_job", "Never_worked", "Private", "Self-employed", "children"]
    )
    residence_type = st.selectbox(
        "Tipe Tempat Tinggal",
        options=["Rural", "Urban"]
    )
    avg_glucose = st.number_input(
        "Rata-rata Kadar Gula Darah (mg/dL)",
        min_value=50.0, max_value=400.0, value=100.0, step=0.1
    )
    bmi = st.number_input(
        "BMI (Indeks Massa Tubuh)",
        min_value=10.0, max_value=100.0, value=25.0, step=0.1
    )
    smoking_status = st.selectbox(
        "Status Merokok",
        options=["Unknown", "formerly smoked", "never smoked", "smokes"]
    )

st.markdown("---")

# tombol prediksi
if st.button("🔍 Prediksi Risiko Stroke", use_container_width=True):

    # MENYAMAKAN FORMAT INPUTAN PENGGUNA
    # Input teks dari pengguna diubah ke angka (encode) menggunakan model encoder yang sama
    gender_enc    = le_gender.transform([gender])[0]
    married_enc   = le_married.transform([ever_married])[0]
    work_enc      = le_work.transform([work_type])[0]
    residence_enc = le_residence.transform([residence_type])[0]
    smoking_enc   = le_smoking.transform([smoking_status])[0]

    hypertension_enc  = 1 if hypertension == "Ya" else 0
    heart_disease_enc = 1 if heart_disease == "Ya" else 0

    # susun data input ke dalam array
    input_data = np.array([[
        gender_enc,
        age,
        hypertension_enc,
        heart_disease_enc,
        married_enc,
        work_enc,
        residence_enc,
        avg_glucose,
        bmi,
        smoking_enc
    ]])

    # SCALING DATA INPUT BARU
    input_scaled = scaler.transform(input_data)

    # MELAKUKAN PREDIKSI BERDASARKAN INPUT
    # model.predict: Menebak hasil kelas (0 / 1)
    # model.predict_proba: Menghitung persentase keyakinan model
    hasil        = model.predict(input_scaled)[0]
    probabilitas = model.predict_proba(input_scaled)[0]

    prob_tidak_stroke = probabilitas[0] * 100
    prob_stroke       = probabilitas[1] * 100

    st.markdown("---")
    st.subheader("📊 Hasil Prediksi")

    if hasil == 1:
        st.error("⚠️ **BERISIKO STROKE**")
        st.markdown(f"Probabilitas risiko stroke: **{prob_stroke:.2f}%**")
        st.markdown(f"Probabilitas tidak stroke: **{prob_tidak_stroke:.2f}%**")
        st.warning(
            "Pasien terindikasi berisiko mengalami stroke. "
            "Disarankan untuk segera berkonsultasi dengan dokter "
            "dan melakukan pemeriksaan lebih lanjut."
        )
    else:
        st.success("✅ **TIDAK BERISIKO STROKE**")
        st.markdown(f"Probabilitas tidak stroke: **{prob_tidak_stroke:.2f}%**")
        st.markdown(f"Probabilitas risiko stroke: **{prob_stroke:.2f}%**")
        st.info(
            "Pasien tidak terindikasi berisiko stroke. "
            "Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin."
        )

# footer
st.markdown("---")
st.markdown(
    "<small>Damar Yudistira Putra | 22.12.2381 | Universitas AMIKOM Yogyakarta | 2026</small>",
    unsafe_allow_html=True
)
