import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import os

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Engine RUL Prediction App")

# Define constants
SEQUENCE_LENGTH = 50
RUL_CLIP_THRESHOLD = 125

# --- Paths (relative to the app's location) ---
# This assumes app.py is in the 'KSA Project' folder
MODEL_PATH = 'lstm_rul_predictor_FINAL.h5'
DATA_DIR = 'CMAPSSData'
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_FD001.txt')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_FD001.txt')
RUL_DATA_PATH = os.path.join(DATA_DIR, 'RUL_FD001.txt')

# --- Data Loading and Caching ---

@st.cache_resource
def load_keras_model():
    """Load the pre-trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is in the same directory as the app.")
        return None
    return load_model(MODEL_PATH)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess all necessary data."""
    # Check for data files
    for path in [TRAIN_DATA_PATH, TEST_DATA_PATH, RUL_DATA_PATH]:
        if not os.path.exists(path):
            st.error(f"Data file not found at {path}. Please ensure the `CMAPSSData` folder is in the same directory as the app.")
            return None, None, None, None, None

    # Define column names
    column_names = ['unit_id', 'time_in_cycles'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]

    # Load data
    train_df = pd.read_csv(TRAIN_DATA_PATH, sep='\\s+', header=None, names=column_names)
    test_df = pd.read_csv(TEST_DATA_PATH, sep='\\s+', header=None, names=column_names)
    rul_df = pd.read_csv(RUL_DATA_PATH, sep='\\s+', header=None, names=['RUL'])

    # Drop constant columns
    constant_columns = ['setting_3'] + [f'sensor_{i}' for i in [1, 5, 6, 10, 16, 18, 19]]
    train_df = train_df.drop(columns=constant_columns)
    test_df = test_df.drop(columns=constant_columns)
    feature_cols = [col for col in train_df.columns if col.startswith('setting') or col.startswith('sensor')]

    # Fit the scaler on the TRAINING data
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

    # Transform test data
    test_df_scaled = test_df.copy()
    test_df_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    # Calculate true RUL for the test set
    max_cycles_test = test_df.groupby('unit_id')['time_in_cycles'].max().reset_index()
    max_cycles_test.columns = ['unit_id', 'max_cycle']
    test_df = pd.merge(test_df, max_cycles_test, on='unit_id', how='left')
    
    # The RUL file gives the final RUL after the last cycle. We need to add it to the max cycle of the test data.
    rul_df['unit_id'] = rul_df.index + 1
    test_df['true_RUL'] = test_df.groupby('unit_id')['time_in_cycles'].transform(max) + rul_df.set_index('unit_id')['RUL']
    test_df['true_RUL'] = test_df['true_RUL'] - test_df['time_in_cycles']
    
    test_df = test_df.drop(columns=['max_cycle'])
    # Clip for consistency with model training
    test_df['true_RUL'] = test_df['true_RUL'].clip(upper=RUL_CLIP_THRESHOLD)


    return test_df, test_df_scaled, scaler, feature_cols, list(test_df['unit_id'].unique())


# --- Main Application ---

# Load model and data
model = load_keras_model()
test_df, test_df_scaled, scaler, feature_cols, unit_ids = load_and_preprocess_data()


st.title("⚙️ Predictive Maintenance: Engine RUL Predictor")

st.markdown("""
This interactive web application demonstrates the final predictive maintenance model.
Select an engine and a specific point in its operational life to predict its Remaining Useful Life (RUL).
""")

if model is None or test_df is None:
    st.warning("Application cannot start due to missing model or data files. Please check the file paths.")
else:
    # --- UI Components ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Engine Selection")
        selected_unit = st.selectbox("Choose an Engine (Unit ID):", unit_ids)

        # Get data for the selected engine
        engine_data = test_df[test_df['unit_id'] == selected_unit].copy()
        max_cycle = engine_data['time_in_cycles'].max()

        # Allow user to select a cycle
        selected_cycle = st.slider(
            "Select an Operational Cycle:",
            min_value=1,
            max_value=int(max_cycle),
            value=max(SEQUENCE_LENGTH, int(max_cycle / 2)), # Default to middle or sequence length
            step=1
        )

    # --- Prediction Logic ---
    with col2:
        st.subheader("Model Prediction")

        # Get the sequence of data up to the selected cycle from the SCALED dataframe
        sequence_data_scaled = test_df_scaled[
            (test_df_scaled['unit_id'] == selected_unit) &
            (test_df_scaled['time_in_cycles'] <= selected_cycle)
        ]
        
        # Ensure we have enough data; pad if necessary
        if len(sequence_data_scaled) < SEQUENCE_LENGTH:
            # Not enough data for a prediction yet
             st.warning(f"Not enough data to make a prediction. At least {SEQUENCE_LENGTH} cycles are needed.")
             predicted_rul = -1
        else:
            # Take the last `SEQUENCE_LENGTH` points
            sequence_to_predict = sequence_data_scaled[feature_cols].tail(SEQUENCE_LENGTH)
            
            # Reshape for the model (data is already scaled)
            input_data = np.array([sequence_to_predict.values])

            # Make a prediction
            prediction = model.predict(input_data)
            predicted_rul = int(prediction[0][0])


        # Get the true RUL for comparison
        true_rul_series = engine_data[engine_data['time_in_cycles'] == selected_cycle]['true_RUL']
        true_rul = int(true_rul_series.values[0]) if not true_rul_series.empty else 0
        
        if predicted_rul != -1:
            st.metric(label="Predicted RUL (Cycles)", value=predicted_rul)
            st.metric(label="True RUL (for comparison)", value=true_rul)
        else:
             st.info("Prediction will appear here once enough cycle data is available.")


    # --- Visualization ---
    st.subheader(f"Sensor Data for Engine {selected_unit}")

    # Select key sensors to plot
    sensors_to_plot = st.multiselect(
        "Select sensors to visualize:",
        options=feature_cols,
        default=[f'sensor_{i}' for i in [2, 3, 4, 7, 11, 12, 15] if f'sensor_{i}' in feature_cols]
    )
    
    if sensors_to_plot:
        # Get data up to the selected cycle for plotting from the ORIGINAL dataframe
        plot_data = engine_data[engine_data['time_in_cycles'] <= selected_cycle]
        
        fig = px.line(
            plot_data,
            x='time_in_cycles',
            y=sensors_to_plot,
            title=f'Sensor Trends up to Cycle {selected_cycle}',
            labels={'time_in_cycles': 'Time in Cycles', 'value': 'Sensor Reading'}
        )
        fig.update_layout(legend_title_text='Sensors')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select one or more sensors to see their trends.")

st.markdown("---")
st.write("Project developed as a portfolio piece for the KSA job market.")
