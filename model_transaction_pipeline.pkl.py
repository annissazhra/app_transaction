import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

# Asumsi kamu sudah memuat dan membersihkan data seperti di UAS.ipynb
# Ganti 'transactions.csv' dengan path file datamu yang sebenarnya jika berbeda
try:
    data = pd.read_csv("transactions.csv")
except FileNotFoundError:
    st.error("❌ File 'transactions.csv' tidak ditemukan. Pastikan file berada di direktori yang sama atau berikan path yang benar.")
    st.stop()

# Mengisi nilai yang hilang (jika ada)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Pembagian data: X adalah fitur, y adalah target (Amount (INR))
X = data.drop('Amount (INR)', axis=1)
y = data['Amount (INR)']

# Membuang kolom yang tidak relevan untuk model
# Pastikan nama kolom ini sesuai dengan data aktualmu
columns_to_drop = ['Transaction ID', 'Timestamp', 'Sender Name', 'Receiver Name']
X = X.drop(columns=columns_to_drop, errors='ignore') # 'errors='ignore' mencegah error jika kolom tidak ada

# Pembagian data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mengidentifikasi fitur kategorikal dan numerik
# Asumsi 'Status', 'Sender UPI ID', 'Receiver UPI ID' adalah kategorikal
categorical_features = ['Status', 'Sender UPI ID', 'Receiver UPI ID']
# Identifikasi fitur numerik yang tersisa secara otomatis
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Pra-pemrosesan: Scaling untuk numerik dan One-Hot Encoding untuk kategorikal
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Biarkan kolom lain yang tidak disebut tetap ada
)

# Inisialisasi model Decision Tree Regressor
# Gunakan parameter terbaik yang kamu temukan (misal: max_depth=3, min_samples_split=5)
# Sesuaikan jika kamu punya parameter terbaik lain
best_dt_model = DecisionTreeRegressor(max_depth=3, min_samples_split=5, random_state=42)

# Membuat pipeline lengkap: preprocessor -> model
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', best_dt_model)
])

# Melatih pipeline pada data pelatihan
full_pipeline.fit(X_train, y_train)

# Menyimpan pipeline yang sudah dilatih (termasuk preprocessor dan model)
joblib.dump(full_pipeline, 'model_transaction_pipeline.pkl')

print("✅ Pipeline model Decision Tree (termasuk preprocessor) berhasil disimpan sebagai 'model_transaction_pipeline.pkl'.")
