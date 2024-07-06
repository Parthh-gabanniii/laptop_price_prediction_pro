import streamlit as st
import pickle
import numpy as np

# Load the model
try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load the DataFrame
try:
    with open('df.pkl', 'rb') as f:
        df = pickle.load(f)
except Exception as e:
    st.error(f"Error loading DataFrame: {e}")

# Add custom CSS for outer border
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f0f5;
    }
    .main .block-container {
        border: 2px solid black;  /* Black border */
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
    }
    h1 {
        color: #ff6347;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>Laptop Predictor</h1>", unsafe_allow_html=True)

if 'df' in locals():
    # Creating columns for layout in 7 horizontal lines
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    col7, col8 = st.columns(2)
    col9, col10 = st.columns(2)
    col11, col12 = st.columns(2)
    col13, col14 = st.columns(2)

    # Assigning each input to a column
    company = col1.selectbox('Brand', df['Company'].unique())
    type = col2.selectbox('Type', df['TypeName'].unique())
    ram = col3.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = col4.number_input('Weight of the Laptop')
    touchscreen = col5.selectbox('Touchscreen', ['No', 'Yes'])
    ips = col6.selectbox('IPS', ['No', 'Yes'])
    screen_size = col7.number_input('Screen Size')
    resolution = col8.selectbox('Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
        '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])
    cpu = col9.selectbox('CPU', df['cpu_brand'].unique())
    hdd = col10.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = col11.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
    gpu = col12.selectbox('GPU', df['Gpu_brand'].unique())
    os = col13.selectbox('OS', df['os'].unique())

    if col14.button('Predict Price'):
        # Query
        ppi = None
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

        query = query.reshape(1, 12)
        try:
            prediction = pipe.predict(query)[0]
            col14.write("Predicted price of the laptop: $", np.exp(prediction))
        except Exception as e:
            col14.error(f"An error occurred: {e}")
else:
    st.error("DataFrame is not available.")
