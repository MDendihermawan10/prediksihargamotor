import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load dataset
df = pd.read_csv('BIKE DETAILS.csv')

# Bersihkan numeric columns
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')
df['ex_showroom_price'] = pd.to_numeric(df['ex_showroom_price'], errors='coerce')
df = df.dropna(subset=['selling_price','year','km_driven','ex_showroom_price'])

# Features
X = df[['name','year','km_driven','seller_type','owner','ex_showroom_price']]
y = df['selling_price']

# Preprocessing
categorical_features = ['name','seller_type','owner']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# Pipeline
model = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
model.fit(X, y)

# Save model compatible dengan Streamlit Cloud
joblib.dump(model, 'motorcycle_model.pkl')
print("✅ Model berhasil dibuat: motorcycle_model.pkl")
