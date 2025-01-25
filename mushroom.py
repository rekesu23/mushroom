import streamlit as st
import numpy as np
import pickle
import os

# Define paths for the model and scaler files
MODEL_PATH = "/full/path/to/random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"

# Function to load the model
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found: {scaler_path}")
        st.stop()

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Load the model and scaler
model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

# Streamlit app
st.title("Mushroom Classification App")
st.write("This app predicts whether a mushroom is edible or poisonous based on its characteristics.")

# Input features
st.sidebar.header("Enter Mushroom Characteristics:")
cap_shape = st.sidebar.selectbox("Cap Shape (0-5)", [0, 1, 2, 3, 4, 5])
cap_surface = st.sidebar.selectbox("Cap Surface (0-3)", [0, 1, 2, 3])
cap_color = st.sidebar.selectbox("Cap Color (0-9)", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
bruises = st.sidebar.selectbox("Bruises (0 or 1)", [0, 1])
odor = st.sidebar.selectbox("Odor (0-8)", [0, 1, 2, 3, 4, 5, 6, 7, 8])
gill_attachment = st.sidebar.selectbox("Gill Attachment (0 or 1)", [0, 1])
gill_spacing = st.sidebar.selectbox("Gill Spacing (0 or 1)", [0, 1])
gill_size = st.sidebar.selectbox("Gill Size (0 or 1)", [0, 1])
gill_color = st.sidebar.selectbox("Gill Color (0-11)", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
stalk_shape = st.sidebar.selectbox("Stalk Shape (0 or 1)", [0, 1])
stalk_root = st.sidebar.selectbox("Stalk Root (0-4)", [0, 1, 2, 3, 4])
stalk_surface_above_ring = st.sidebar.selectbox("Stalk Surface Above Ring (0-3)", [0, 1, 2, 3])
stalk_surface_below_ring = st.sidebar.selectbox("Stalk Surface Below Ring (0-3)", [0, 1, 2, 3])
stalk_color_above_ring = st.sidebar.selectbox("Stalk Color Above Ring (0-8)", [0, 1, 2, 3, 4, 5, 6, 7, 8])
stalk_color_below_ring = st.sidebar.selectbox("Stalk Color Below Ring (0-8)", [0, 1, 2, 3, 4, 5, 6, 7, 8])
veil_type = st.sidebar.selectbox("Veil Type (0)", [0])
veil_color = st.sidebar.selectbox("Veil Color (0-3)", [0, 1, 2, 3])
ring_number = st.sidebar.selectbox("Ring Number (0-2)", [0, 1, 2])
ring_type = st.sidebar.selectbox("Ring Type (0-4)", [0, 1, 2, 3, 4])
spore_print_color = st.sidebar.selectbox("Spore Print Color (0-8)", [0, 1, 2, 3, 4, 5, 6, 7, 8])
population = st.sidebar.selectbox("Population (0-5)", [0, 1, 2, 3, 4, 5])
habitat = st.sidebar.selectbox("Habitat (0-6)", [0, 1, 2, 3, 4, 5, 6])

# Combine inputs into a single array
input_features = np.array([cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, 
                           gill_spacing, gill_size, gill_color, stalk_shape, stalk_root, 
                           stalk_surface_above_ring, stalk_surface_below_ring, 
                           stalk_color_above_ring, stalk_color_below_ring, veil_type, 
                           veil_color, ring_number, ring_type, spore_print_color, 
                           population, habitat]).reshape(1, -1)

# Scale the input features
input_features_scaled = scaler.transform(input_features)

# Make a prediction
prediction = model.predict(input_features_scaled)
probability = model.predict_proba(input_features_scaled)[0, 1]

# Display the results
st.subheader("Prediction Result:")
if prediction[0] == 0:
    st.success(f"The mushroom is **Edible**. (Probability: {1 - probability:.2f})")
else:
    st.error(f"The mushroom is **Poisonous**. (Probability: {probability:.2f})")
