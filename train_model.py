import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib
import os

# --- Load dataset ---
file_path = 'BIKE DETAILS.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset '{file_path}' tidak ditemukan. Pastikan ada di folder project.")

df = pd.read_csv(file_path)

# --- Clean numeric columns ---
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')
df['ex_showroom_price'] = pd.to_numeric(df['ex_showroom_price'], errors='coerce')

# Drop rows invalid
df = df.dropna(subset=['selling_price','year','km_driven','ex_showroom_price'])

print("Data valid:", len(df))

# --- Features & target
X = df[['name','year','km_driven','seller_type','owner','ex_showroom_price']]
y = df['selling_price']

# --- Preprocessing categorical
cat_cols = ['name','seller_type','owner']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
    remainder='passthrough'
)

model = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42))
])

# --- Train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model.fit(X_train, y_train)

# --- Evaluasi
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
print("Mean Squared Error:", mse)

# --- Save model
joblib.dump(model, 'motorcycle_model.pkl')
print("Model tersimpan: motorcycle_model.pkl")
