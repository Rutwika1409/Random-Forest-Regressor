import streamlit as st
import pandas as pd
import joblib

model = joblib.load(
    'models/random_forest_model.pkl'
)

encoders = joblib.load(
    'models/encoders.pkl'
)

features = joblib.load(
    'models/features.pkl'
)

st.set_page_config(
    page_title='Car Price Prediction',
    layout='centered'
)

st.title('Car Price Prediction')

st.write('Random Forest Regressor')

# ---------------- INPUTS ----------------

car_name = st.selectbox(
    'Car Name',
    encoders['car_name'].classes_
)

brand = st.selectbox(
    'Brand',
    encoders['brand'].classes_
)

model_name = st.selectbox(
    'Model',
    encoders['model'].classes_
)

vehicle_age = st.slider(
    'Vehicle Age',
    0,
    20,
    5
)

km_driven = st.number_input(
    'KM Driven',
    0,
    500000,
    30000
)

seller_type = st.radio(
    'Seller Type',
    encoders['seller_type'].classes_
)

fuel_type = st.radio(
    'Fuel Type',
    encoders['fuel_type'].classes_
)

transmission_type = st.radio(
    'Transmission Type',
    encoders['transmission_type'].classes_
)

mileage = st.slider(
    'Mileage',
    5.0,
    40.0,
    18.0
)

engine = st.number_input(
    'Engine CC',
    500,
    5000,
    1200
)

max_power = st.slider(
    'Max Power',
    20.0,
    500.0,
    80.0
)

seats = st.slider(
    'Seats',
    2,
    10,
    5
)

# ---------------- ENCODING ----------------

input_data = pd.DataFrame([{

    'car_name': encoders['car_name'].transform([car_name])[0],

    'brand': encoders['brand'].transform([brand])[0],

    'model': encoders['model'].transform([model_name])[0],

    'vehicle_age': vehicle_age,

    'km_driven': km_driven,

    'seller_type': encoders['seller_type'].transform([seller_type])[0],

    'fuel_type': encoders['fuel_type'].transform([fuel_type])[0],

    'transmission_type': encoders['transmission_type'].transform([transmission_type])[0],

    'mileage': mileage,

    'engine': engine,

    'max_power': max_power,

    'seats': seats
}])

input_data = input_data[features]

# ---------------- PREDICTION ----------------

if st.button('Predict Price'):

    prediction = model.predict(input_data)[0]

    st.success(
        f'Estimated Car Price : ₹ {prediction:,.2f}'
    )