import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data (ganti 'data.csv' dengan nama file dataset Anda)
data = pd.read_csv('./smartphone_hasil_preprocessing.csv')

# Memisahkan fitur dan target
X = data.drop('price', axis=1).drop('avg_rating', axis=1)
y = data['price']

# Mengubah data non-numerik menjadi numerik menggunakan One-Hot Encoding
X = pd.get_dummies(X)

# Membangun model regresi
model = LinearRegression()
model.fit(X, y)

# Membuat fungsi prediksi harga handphone
def predict_price(features):
    features_encoded = pd.get_dummies(features)
    # Menambahkan kolom missing jika ada fitur yang tidak ada pada dataset asli
    features_encoded = features_encoded.reindex(columns=X.columns, fill_value=0)
    price_prediction = model.predict(features_encoded)  # Remove the outer list
    return price_prediction[0]

# Membuat aplikasi web dengan Streamlit
st.title('Prediksi Harga Handphone')
st.write('Masukkan kombinasi fitur-fitur handphone untuk memprediksi harga:')

# Input fitur-fitur handphone
# Input fitur-fitur handphone
brand_names = data['brand_name'].unique()
brand_name = st.selectbox('Brand Name:', brand_names)

# Filter model_name berdasarkan brand_name yang dipilih
models_for_selected_brand = data[data['brand_name'] == brand_name]['model'].unique()
model_name = st.selectbox('Model Name:', models_for_selected_brand)

is_5G = st.checkbox('5G Support')
is_fast_charging_available = st.checkbox('Fast Charging Available')

processor_brands = data['processor_brand'].unique()
processor_brand = st.selectbox('Processor Brand:', processor_brands)

num_cores_options = sorted(data['num_cores'].unique())
num_cores = st.selectbox('Number of Cores:', num_cores_options)

processor_speed_options = sorted(data['processor_speed'].unique())
processor_speed = st.selectbox('Processor Speed (GHz):', processor_speed_options)

battery_capacity_options = sorted(data['battery_capacity'].unique())
battery_capacity = st.selectbox('Battery Capacity (mAh):', battery_capacity_options)

fast_charging_speed_options = sorted(data['fast_charging'].unique())
fast_charging_speed = st.selectbox('Fast Charging Speed (W):', fast_charging_speed_options)

ram_capacity_options = sorted(data['ram_capacity'].unique())
ram_capacity = st.selectbox('RAM Capacity (GB):', ram_capacity_options)

internal_memory_options = sorted(data['internal_memory'].unique())
internal_memory = st.selectbox('Internal Memory (GB):', internal_memory_options)

screen_size_options = sorted(data['screen_size'].unique())
screen_size = st.selectbox('Screen Size (inch):', screen_size_options)

refresh_rate_options = sorted(data['refresh_rate'].unique())
refresh_rate = st.selectbox('Refresh Rate (Hz):', refresh_rate_options)

num_rear_cameras_options = sorted(data['num_rear_cameras'].unique())
num_rear_cameras = st.selectbox('Number of Rear Cameras:', num_rear_cameras_options)


os = st.selectbox('Operating System', ['Android', 'iOS'])

primary_camera_rear_options = sorted(data['primary_camera_rear'].unique())
primary_camera_rear = st.selectbox('Primary Rear Camera (MP):', primary_camera_rear_options)

primary_camera_front_options = sorted(data['primary_camera_front'].unique())
primary_camera_front = st.selectbox('Primary Front Camera (MP):', primary_camera_front_options)

is_extended_memory_available = st.checkbox('Extended Memory Available')

resolutions_height = data['resolution_height'].unique()
resolution_height = st.selectbox('Resolution Height (pixels):', resolutions_height)

resolutions_width = data['resolution_width'].unique()
resolution_width = st.selectbox('Resolution Width (pixels):', resolutions_width)

# Prediksi harga handphone berdasarkan input fitur-fitur
input_features = pd.DataFrame({
    '5G_or_not': [1 if is_5G else 0],
    'num_cores': [num_cores],
    'processor_speed': [processor_speed],
    'battery_capacity': [battery_capacity],
    'fast_charging_available': [1 if is_fast_charging_available else 0],
    'fast_charging': [fast_charging_speed],
    'ram_capacity': [ram_capacity],
    'internal_memory': [internal_memory],
    'screen_size': [screen_size],
    'refresh_rate': [refresh_rate],
    'num_rear_cameras': [num_rear_cameras],
    'os_Android': [1 if os == 'Android' else 0],
    'os_iOS': [1 if os == 'iOS' else 0],
    'primary_camera_rear': [primary_camera_rear],
    'primary_camera_front': [primary_camera_front],
    'extended_memory_available': [1 if is_extended_memory_available else 0],
    'resolution_height': [resolution_height],
    'resolution_width': [resolution_width]
})

# Ensure data types are consistent with the model
input_features = input_features.astype(float)

if st.button('Predict'):
    # Handle missing values
    input_features = input_features.fillna(0)
    
    predicted_price = predict_price(input_features)

    # Convert predicted price to a more readable format with two decimal places
    predicted_price = predict_price(input_features) / 100  # Divide by 100 to convert cents to dollars
    st.write(f'Predicted Harga: ${predicted_price:.2f}')