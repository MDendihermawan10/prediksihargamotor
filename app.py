import subprocess
import sys
import os

# ==========================
# Auto-install joblib
# ==========================
try:
    import joblib
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib

# ==========================
# Import lain
# ==========================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# Cek model dan dataset
# ==========================
model_file = 'motorcycle_model.pkl'
data_file = 'BIKE DETAILS.csv'

if not os.path.exists(model_file):
    st.error("Model belum dibuat. Jalankan train_model.py dulu.")
    st.stop()
if not os.path.exists(data_file):
    st.error("Dataset tidak ditemukan.")
    st.stop()

# ==========================
# Load model & dataset
# ==========================
model = joblib.load(model_file)
df = pd.read_csv(data_file)

# Bersihkan numeric columns
numeric_cols = ['selling_price','year','km_driven','ex_showroom_price']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=numeric_cols)
df[numeric_cols] = df[numeric_cols].astype(float)

# ==========================
# Style Streamlit
# ==========================
st.set_page_config(page_title="Prediksi Harga Motor Bekas", layout="wide")
st.title("🏍️ Prediksi Harga Motor Bekas (Random Forest)")
st.markdown("Gunakan aplikasi ini untuk memprediksi harga motor bekas berdasarkan data aktual.")

# ==========================
# Ringkasan Dataset
# ==========================
with st.expander("📊 Ringkasan Dataset"):
    st.write(f"Jumlah data valid: {len(df)}")
    st.write(df[numeric_cols].describe())

    # Histogram harga dan mileage
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].hist(df['selling_price'], bins=20, color='skyblue', edgecolor='black')
    ax[0].set_title("Distribusi Harga Jual")
    ax[0].set_xlabel("Harga")
    ax[0].set_ylabel("Jumlah")

    ax[1].hist(df['km_driven'], bins=20, color='lightgreen', edgecolor='black')
    ax[1].set_title("Distribusi Jarak Tempuh (Km Driven)")
    ax[1].set_xlabel("Km Driven")
    ax[1].set_ylabel("Jumlah")

    st.pyplot(fig)

# ==========================
# Prediksi Single Motor
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

if st.button("Prediksi Harga"):
    input_df = pd.DataFrame([{
        'name': name,
        'year': year,
        'km_driven': km,
        'seller_type': seller_type,
        'owner': owner,
        'ex_showroom_price': ex_price
    }])
    try:
        pred = model.predict(input_df)[0]
        harga_rupiah = f"Rp {int(pred):,}".replace(",", ".")
        st.success(f"💰 Estimasi Harga Motor Bekas: {harga_rupiah}")
    except Exception as e:
        st.error(f"Prediksi gagal: {e}")

# ==========================
# Feature Importance
# ==========================
st.subheader("🏆 Feature Importance (Top 10)")
try:
    if hasattr(model, 'named_steps') and 'rf' in model.named_steps:
        rf_model = model.named_steps['rf']
        prep_step = model.named_steps.get('prep', None)
        if prep_step:
            feature_names = prep_step.transformers_[0][1].get_feature_names_out()
            feature_names = list(feature_names) + ['year','km_driven','ex_showroom_price']
        else:
            feature_names = ['year','km_driven','ex_showroom_price']
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        fi_df = fi_df.sort_values(by='importance', ascending=False).head(10)
        st.bar_chart(fi_df.set_index('feature'))
    else:
        st.info("Feature importance tidak tersedia untuk model ini.")
except Exception as e:
    st.info("Feature importance tidak tersedia: " + str(e))

# ==========================
# Scatter Plot Harga vs Km Driven
# ==========================
st.subheader("📈 Scatter Plot Harga vs Jarak Tempuh")
st.scatter_chart(df[['km_driven','selling_price']].rename(columns={'km_driven':'x','selling_price':'y'}))
