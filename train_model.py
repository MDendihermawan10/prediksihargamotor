import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def main():
    data_file = 'BIKE_DETAILS.csv'
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"File '{data_file}' tidak ditemukan. Letakkan CSV di folder yang sama.")

    df = pd.read_csv(data_file)
    numeric_cols = ['selling_price', 'year', 'km_driven', 'ex_showroom_price']
    for col in numeric_cols:
        if col not in df.columns:
            raise KeyError(f"Kolom '{col}' tidak ditemukan di dataset!")
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_cols)

    X = df[['name', 'year', 'km_driven', 'seller_type', 'owner', 'ex_showroom_price']]
    y = df['selling_price']

    categorical_features = ['name', 'seller_type', 'owner']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
        remainder='passthrough'
    )

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    print("Melatih model Random Forest...")
    model_pipeline.fit(X, y)
    joblib.dump(model_pipeline, 'motorcycle_model.pkl')
    print("✅ Model berhasil dibuat: motorcycle_model.pkl")

if __name__ == "__main__":
    main()
