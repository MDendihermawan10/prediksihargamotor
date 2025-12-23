import os
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ==========================
# Files
# ==========================
model_file = 'motorcycle_model.pkl'
data_file = 'BIKE_DETAILS.csv'

# ==========================
# Cek file
# ==========================
if not os.path.exists(model_file):
    st.error("Model belum dibuat. Jalankan train_model.py dulu."); st.stop()
if not os.path.exists(data_file):
    st.error("Dataset tidak ditemukan."); st.stop()

# ==========================
# Load model & data
# ==========================
model = joblib.load(model_file)
df = pd.read_csv(data_file)
numeric_cols = ['selling_price','year','km_driven','ex_showroom_price']
for col in numeric_cols: 
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=numeric_cols)
df[numeric_cols] = df[numeric_cols].astype(float)

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Prediksi Harga Motor Bekas", layout="wide")
st.title("🏍️ Prediksi Harga Motor Bekas (Random Forest)")
st.markdown("""
Selamat datang! Gunakan panel input di bawah untuk memprediksi harga motor bekas berdasarkan fitur yang tersedia.
""")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ Informasi Dataset")
    st.write(f"Jumlah data: {len(df)} motor")
    st.write(f"Rentang tahun: {int(df['year'].min())} - {int(df['year'].max())}")
    st.write(f"Rentang harga: Rp {int(df['selling_price'].min()):,} - Rp {int(df['selling_price'].max()):,}".replace(",", "."))
    if st.checkbox("Tampilkan dataset sample"):
        st.dataframe(df.head(10))

# ==========================
# Input user
# ==========================
st.subheader("💡 Prediksi Harga Motor")
col1, col2 = st.columns(2)
with col1:
    name = st.selectbox("Nama Motor", df['name'].unique())
    year = st.number_input("Tahun", int(df['year'].min()), int(df['year'].max()), int(df['year'].median()))
    km = st.number_input("Km Driven", 0, int(df['km_driven'].max()), int(df['km_driven'].median()))
with col2:
    seller_type = st.selectbox("Seller Type", df['seller_type'].dropna().unique())
    owner = st.selectbox("Owner", df['owner'].dropna().unique())
    ex_price = st.number_input("Ex Showroom Price", 0, int(df['ex_showroom_price'].max()), int(df['ex_showroom_price'].median()))

# ==========================
# Prediksi
# ==========================
if st.button("Prediksi Harga"):
    input_df = pd.DataFrame([{
        'name': name, 'year': year, 'km_driven': km,
        'seller_type': seller_type, 'owner': owner, 'ex_showroom_price': ex_price
    }])
    pred = model.predict(input_df)[0]
    st.metric(label="💰 Estimasi Harga Motor Bekas", value=f"Rp {int(pred):,}".replace(",", "."))

    st.markdown("---")
    st.subheader("📊 Visualisasi Harga Motor dari Dataset")
    
    # Histogram harga
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df['selling_price'], bins=20, color='#4CAF50', edgecolor='black')
    ax.axvline(pred, color='red', linestyle='--', label='Prediksi Anda')
    ax.set_xlabel("Harga (Rp)")
    ax.set_ylabel("Jumlah Motor")
    ax.set_title("Distribusi Harga Motor Bekas")
    ax.legend()
    st.pyplot(fig)
