import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

# ==========================
# Cek keberadaan file CSV
# ==========================
data_file = 'BIKE_DETAILS.csv'
if not os.path.exists(data_file):
    raise FileNotFoundError(
        f"File '{data_file}' tidak ditemukan. Letakkan file CSV di folder yang sama dengan script ini."
    )

# ==========================
# Load dataset
# ==========================
df = pd.read_csv(data_file)

# ==========================
# Bersihkan numeric columns
# ==========================
numeric_cols = ['selling_price', 'year', 'km_driven', 'ex_showroom_price']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=numeric_cols)

# ==========================
# Features & target
# ==========================
X = df[['name', 'year', 'km_driven', 'seller_type', 'owner', 'ex_showroom_price']]
y = df['selling_price']

# ==========================
# Preprocessing pipeline
# ==========================
categorical_features = ['name', 'seller_type', 'owner']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'  # numeric columns tetap dipertahankan
)

# ==========================
# Pipeline model
# ==========================
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# ==========================
# Train model
# ==========================
model_pipeline.fit(X, y)

# ==========================
# Save model
# ==========================
model_filename = 'motorcycle_model.pkl'
joblib.dump(model_pipeline, model_filename)
print(f"✅ Model berhasil dibuat: {model_filename}")
