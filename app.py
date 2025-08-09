# ============================================================
# 💰 APLIKASI PREDIKSI JUMLAH TRANSAKSI - DECISION TREE
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import os

# ============================================================
# 1️⃣ Load Model dan Preprocessor
# ============================================================
@st.cache_resource
def load_assets():
    """
    Memuat model Decision Tree dan preprocessor pipeline yang sudah terlatih.
    """
    model_path = "model_decision_tree.pkl"
    preprocessor_path = "preprocessor_pipeline.pkl"

    if not os.path.exists(model_path):
        st.error(f"❌ File model '{model_path}' tidak ditemukan. Mohon buat file tersebut terlebih dahulu.")
        st.stop()
    if not os.path.exists(preprocessor_path):
        st.error(f"❌ File preprocessor '{preprocessor_path}' tidak ditemukan. Mohon buat file tersebut terlebih dahulu.")
        st.stop()

    try:
        # Memuat model yang sudah dilatih
        trained_model = joblib.load(model_path)
        # Memuat preprocessor pipeline yang sudah dilatih
        preprocessor = joblib.load(preprocessor_path)
        return trained_model, preprocessor
    except Exception as e:
        st.error(f"❌ Gagal memuat aset: {e}. Pastikan versi pustaka Anda konsisten.")
        st.stop()

model, preprocessor = load_assets()

# ============================================================
# 2️⃣ Konfigurasi Aplikasi
# ============================================================
st.set_page_config(
    page_title="Prediksi Jumlah Transaksi",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Aplikasi Prediksi Jumlah Transaksi (Decision Tree)")
st.markdown("Masukkan detail transaksi untuk memprediksi jumlahnya.")

# ============================================================
# 3️⃣ Formulir Input
# ============================================================
with st.form("transaction_amount_form"):
    st.subheader("📝 Masukkan Detail Transaksi")

    # Input untuk fitur kategorikal
    # Asumsi status memiliki nilai-nilai ini. Kamu bisa menambahkannya jika ada nilai lain
    status = st.selectbox("Status Transaksi", ('Success', 'Failed', 'Pending', 'Cancelled'))
    sender_upi_id = st.text_input("Sender UPI ID (contoh: user123@upi)")
    receiver_upi_id = st.text_input("Receiver UPI ID (contoh: merchantabc@upi)")

    submitted = st.form_submit_button("🔮 Prediksi Jumlah")

    if submitted:
        try:
            # Buat DataFrame dari input pengguna
            # Pastikan nama kolom sesuai dengan yang digunakan saat pelatihan preprocessor
            input_data = pd.DataFrame([{
                'Status': status,
                'Sender UPI ID': sender_upi_id,
                'Receiver UPI ID': receiver_upi_id
            }])

            # Pra-pemrosesan input data menggunakan preprocessor yang sudah dilatih
            # Penting: preprocessor.transform() akan menghasilkan sparse matrix jika ada OneHotEncoder,
            # pastikan model bisa menerima format itu atau ubah ke dense array jika perlu.
            # Decision Tree biasanya bisa menerima sparse.
            processed_input_data = preprocessor.transform(input_data)

            # Melakukan prediksi dengan model
            prediction = model.predict(processed_input_data)

            st.subheader("✅ Prediksi Berhasil!")
            st.success(f"Jumlah transaksi yang diprediksi adalah: ₹{prediction[0]:,.2f}")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
