import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
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

    # preprocessing
    df.drop(columns=['id'], inplace=True)
    df = df[df['gender'] != 'Other']

    # isi missing value bmi dengan median
    bmi_median = df['bmi'].median()
    df['bmi'].fillna(bmi_median, inplace=True)

    # pastikan tidak ada nilai kosong yang tersisa sebelum SMOTE
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['avg_glucose_level'] = pd.to_numeric(df['avg_glucose_level'], errors='coerce')
    df = df.fillna(df.median(numeric_only=True))
    df = df.dropna()

    # label encoding kolom kategorikal
    le_gender         = LabelEncoder()
    le_married        = LabelEncoder()
    le_work           = LabelEncoder()
    le_residence      = LabelEncoder()
    le_smoking        = LabelEncoder()

    df['gender']         = le_gender.fit_transform(df['gender'])
    df['ever_married']   = le_married.fit_transform(df['ever_married'])
    df['work_type']      = le_work.fit_transform(df['work_type'])
    df['Residence_type'] = le_residence.fit_transform(df['Residence_type'])
    df['smoking_status'] = le_smoking.fit_transform(df['smoking_status'])

    # pisahkan fitur dan label
    X = df.drop(columns=['stroke'])
    y = df['stroke']

    # tangani imbalanced data dengan SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # split data 80:20
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # normalisasi data
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    # latih model dengan parameter terbaik hasil GridSearchCV
    model = RandomForestClassifier(
        n_estimators=200,
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

    # encode input sesuai label encoder yang sudah dilatih
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

    # normalisasi input
    input_scaled = scaler.transform(input_data)

    # prediksi
    hasil       = model.predict(input_scaled)[0]
    probabilitas = model.predict_proba(input_scaled)[0]

    prob_tidak_stroke = probabilitas[0] * 100
    prob_stroke       = probabilitas[1] * 100

    st.markdown("---")
    st.subheader("📊 Hasil Prediksi")

    if hasil == 1:
        st.error(f"⚠️ **BERISIKO STROKE**")
        st.markdown(f"Probabilitas risiko stroke: **{prob_stroke:.2f}%**")
        st.markdown(f"Probabilitas tidak stroke: **{prob_tidak_stroke:.2f}%**")
        st.warning(
            "Pasien terindikasi berisiko mengalami stroke. "
            "Disarankan untuk segera berkonsultasi dengan dokter "
            "dan melakukan pemeriksaan lebih lanjut."
        )
    else:
        st.success(f"✅ **TIDAK BERISIKO STROKE**")
        st.markdown(f"Probabilitas tidak stroke: **{prob_tidak_stroke:.2f}%**")
        st.markdown(f"Probabilitas risiko stroke: **{prob_stroke:.2f}%**")
        st.info(
            "Pasien tidak terindikasi berisiko stroke. "
            "Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin."
        )

    # progress bar probabilitas
    st.markdown("**Tingkat Risiko Stroke:**")
    st.progress(int(prob_stroke))
    st.markdown(f"*{prob_stroke:.2f}% kemungkinan stroke*")

# footer
st.markdown("---")
st.markdown(
    "<small>Damar Yudistira Putra | 22.12.2381 | Universitas AMIKOM Yogyakarta | 2026</small>",
    unsafe_allow_html=True
)
