import subprocess, sys, os
try:
    import joblib
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

model_file = 'motorcycle_model.pkl'
data_file = 'BIKE DETAILS.csv'
if not os.path.exists(model_file):
    st.error("Model belum dibuat. Jalankan train_model.py dulu."); st.stop()
if not os.path.exists(data_file):
    st.error("Dataset tidak ditemukan."); st.stop()

model = joblib.load(model_file)
df = pd.read_csv(data_file)
numeric_cols = ['selling_price','year','km_driven','ex_showroom_price']
for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=numeric_cols)
df[numeric_cols] = df[numeric_cols].astype(float)

st.set_page_config(page_title="Prediksi Harga Motor Bekas", layout="wide")
st.title("🏍️ Prediksi Harga Motor Bekas (Random Forest)")

st.subheader("💡 Prediksi Harga Motor")
col1,col2 = st.columns(2)
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
        'name': name,'year': year,'km_driven': km,
        'seller_type': seller_type,'owner': owner,'ex_showroom_price': ex_price
    }])
    pred = model.predict(input_df)[0]
    st.success(f"💰 Estimasi Harga Motor Bekas: Rp {int(pred):,}".replace(",", "."))
