import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import os

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Engine RUL Prediction App")

# --- Constants ---
SEQUENCE_LENGTH = 50
RUL_CLIP_THRESHOLD = 125
FEATURE_COLS = ['setting_1', 'setting_2', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
                'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
                'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']

# --- Paths (relative to the app's location) ---
MODEL_PATH = os.path.join('models', 'lstm_rul_predictor_FINAL.h5')
DATA_DIR = 'data'
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_FD001.txt')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_FD001.txt')
RUL_DATA_PATH = os.path.join(DATA_DIR, 'RUL_FD001.txt')

# --- Data Loading and Caching ---

@st.cache_resource
def load_keras_model():
    """Load the pre-trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure it's in the 'models' subfolder.")
        return None
    return load_model(MODEL_PATH)

@st.cache_data
def load_all_data_and_scaler():
    """Load and preprocess all necessary data and fit the scaler."""
    for path in [TRAIN_DATA_PATH, TEST_DATA_PATH, RUL_DATA_PATH]:
        if not os.path.exists(path):
            st.error(f"Data file not found at {path}. Please ensure it's in the 'data' subfolder.")
            return None, None, None, None

    column_names = ['unit_id', 'time_in_cycles'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    
    # --- Fit Scaler on Training Data (CRITICAL) ---
    train_df_for_scaler = pd.read_csv(TRAIN_DATA_PATH, sep='\\s+', header=None, names=column_names)
    scaler = MinMaxScaler()
    scaler.fit(train_df_for_scaler[FEATURE_COLS])

    # --- Process Test Data for Demo ---
    test_df = pd.read_csv(TEST_DATA_PATH, sep='\\s+', header=None, names=column_names)
    rul_df = pd.read_csv(RUL_DATA_PATH, sep='\\s+', header=None, names=['RUL_at_end'])
    rul_df['unit_id'] = rul_df.index + 1

    # Robust True RUL Calculation
    test_df = pd.merge(test_df, rul_df, on='unit_id', how='left')
    max_cycles_test = test_df.groupby('unit_id')['time_in_cycles'].transform(max)
    test_df['true_RUL'] = max_cycles_test + test_df['RUL_at_end'] - test_df['time_in_cycles']
    test_df = test_df.drop(columns=['RUL_at_end'])
    test_df['true_RUL'] = test_df['true_RUL'].clip(upper=RUL_CLIP_THRESHOLD)
    
    test_df_scaled = test_df.copy()
    test_df_scaled[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])
    
    unit_ids = list(test_df['unit_id'].unique())
    
    return test_df, test_df_scaled, scaler, unit_ids

# --- Main Application ---
st.title("⚙️ Predictive Maintenance: Engine RUL Predictor")

model = load_keras_model()
test_df, test_df_scaled, scaler, unit_ids = load_all_data_and_scaler()

if model is None or test_df is None or scaler is None:
    st.error("Application cannot start due to missing model, data, or scaler. Please check file paths.")
else:
    tab1, tab2, tab3 = st.tabs(["📊 Demo on Test Data", "🚀 Predict on Your Data", "🧠 About the Project"])

    # --- TAB 1: DEMO ON TEST DATA ---
    with tab1:
        st.header("Predict RUL for Engines from the Test Dataset")
        st.info("This tab lets you explore the model's performance on the original test data from the competition.")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Engine Selection")
            selected_unit = st.selectbox("Choose an Engine (Unit ID):", unit_ids)
            engine_data = test_df[test_df['unit_id'] == selected_unit].copy()
            max_cycle = engine_data['time_in_cycles'].max()

            selected_cycle = st.slider(
                "Select an Operational Cycle:",
                min_value=1, max_value=int(max_cycle),
                value=max(SEQUENCE_LENGTH, int(max_cycle / 2)),
                step=1,
                help="Slide to a point in the engine's life to predict its future."
            )

        with col2:
            st.subheader("Model Prediction")
            sequence_data_scaled = test_df_scaled[
                (test_df_scaled['unit_id'] == selected_unit) &
                (test_df_scaled['time_in_cycles'] <= selected_cycle)
            ]
            
            predicted_rul = -1
            if len(sequence_data_scaled) < SEQUENCE_LENGTH:
                 st.warning(f"Not enough data. At least {SEQUENCE_LENGTH} cycles are needed for a prediction.")
            else:
                sequence_to_predict = sequence_data_scaled[FEATURE_COLS].tail(SEQUENCE_LENGTH)
                input_data = np.array([sequence_to_predict.values])
                prediction = model.predict(input_data)
                predicted_rul = int(prediction[0][0])

            true_rul_series = engine_data[engine_data['time_in_cycles'] == selected_cycle]['true_RUL']
            # Safety check for NaN before converting to int
            if not true_rul_series.empty and pd.notna(true_rul_series.values[0]):
                true_rul = int(true_rul_series.values[0])
            else:
                true_rul = 0 
            
            if predicted_rul != -1:
                st.metric(label="Predicted RUL (Cycles)", value=predicted_rul)
                st.metric(label="True RUL (for comparison)", value=true_rul, delta=f"{predicted_rul - true_rul}", delta_color="inverse")
            else:
                 st.info("Prediction will appear here once enough cycle data is available.")

        st.subheader(f"Sensor Data for Engine {selected_unit}")
        sensors_to_plot = st.multiselect(
            "Select sensors to visualize:",
            options=FEATURE_COLS,
            default=[f'sensor_{i}' for i in [2, 3, 4, 7, 11, 12, 15] if f'sensor_{i}' in FEATURE_COLS]
        )
        if sensors_to_plot:
            plot_data = engine_data[engine_data['time_in_cycles'] <= selected_cycle]
            fig = px.line(
                plot_data, x='time_in_cycles', y=sensors_to_plot,
                title=f'Sensor Trends up to Cycle {selected_cycle}',
                labels={'time_in_cycles': 'Time in Cycles', 'value': 'Sensor Reading'}
            )
            fig.update_layout(legend_title_text='Sensors')
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: PREDICT ON YOUR DATA ---
    with tab2:
        st.header("Upload Your Engine's Time-Series Data")
        st.info("Upload a CSV file of your own engine data to get a live RUL prediction.")
        uploaded_file = st.file_uploader(
            "Upload a CSV file with your engine's sensor data.",
            type="csv",
            help=f"File must contain at least {SEQUENCE_LENGTH} rows and the columns: {', '.join(FEATURE_COLS)}"
        )
        
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                # --- Validation and Prediction ---
                if len(new_df) < SEQUENCE_LENGTH:
                    st.error(f"Error: Not enough data. The model requires at least {SEQUENCE_LENGTH} rows, but your file only has {len(new_df)}.")
                elif not all(col in new_df.columns for col in FEATURE_COLS):
                    missing_cols = [col for col in FEATURE_COLS if col not in new_df.columns]
                    st.error(f"Error: The uploaded file is missing the following required columns: {', '.join(missing_cols)}")
                else:
                    st.subheader("Prediction Result")
                    with st.spinner("Processing data and making prediction..."):
                        sequence_to_predict = new_df[FEATURE_COLS].tail(SEQUENCE_LENGTH)
                        scaled_sequence = scaler.transform(sequence_to_predict)
                        input_data = np.reshape(scaled_sequence, (1, SEQUENCE_LENGTH, len(FEATURE_COLS)))
                        
                        prediction = model.predict(input_data)
                        predicted_rul = int(prediction[0][0])
                        
                        st.metric(label="Predicted RUL for Your Engine (Cycles)", value=predicted_rul)
                        
                        st.subheader("Data Used for Prediction")
                        st.write(f"The prediction was based on the last {SEQUENCE_LENGTH} rows of your data:")
                        st.dataframe(sequence_to_predict)
                        
                        # --- Add Visualization for Uploaded Data ---
                        st.subheader("Visualize Your Uploaded Data")
                        if 'time_in_cycles' not in new_df.columns:
                            st.warning("Cannot create visualization because the uploaded file is missing a 'time_in_cycles' column.")
                        else:
                            sensors_to_plot_uploaded = st.multiselect(
                                "Select sensors from your file to visualize:",
                                options=FEATURE_COLS,
                                default=[f'sensor_{i}' for i in [2, 3, 4, 7, 11, 12, 15] if f'sensor_{i}' in FEATURE_COLS],
                                key="uploaded_sensors_multiselect" # Unique key for this widget
                            )
                            if sensors_to_plot_uploaded:
                                fig_uploaded = px.line(
                                    new_df, x='time_in_cycles', y=sensors_to_plot_uploaded,
                                    title='Sensor Trends from Your Uploaded File',
                                    labels={'time_in_cycles': 'Time in Cycles', 'value': 'Sensor Reading'}
                                )
                                fig_uploaded.update_layout(legend_title_text='Sensors')
                                st.plotly_chart(fig_uploaded, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

    # --- TAB 3: ABOUT THE PROJECT ---
    with tab3:
        st.header("About the Project")
        st.markdown("""
        This interactive web application is a portfolio project designed to showcase an end-to-end machine learning workflow for a real-world industrial problem. It is particularly relevant for roles in data science and machine learning within Saudi Arabia's growing industrial and energy sectors, aligning with Vision 2030.

        ### Project Goal
        The primary objective is to predict the **Remaining Useful Life (RUL)** of an aircraft turbofan engine. RUL is the number of operational cycles an engine has left before it is likely to fail. Accurate RUL prediction is the cornerstone of **predictive maintenance**, which allows for:
        -   Scheduling maintenance proactively before failure occurs.
        -   Minimizing costly unplanned downtime.
        -   Maximizing the operational life of expensive assets.

        ### Model Details
        -   **Model Type:** Long Short-Term Memory (LSTM) Neural Network.
        -   **Why LSTM?** LSTMs are a special type of Recurrent Neural Network (RNN) that are exceptionally good at learning from time-series data, as they can remember patterns over long sequences of sensor readings.
        -   **Final Performance (RMSE):** **10.96 cycles**. This means that, on average, the model's prediction is only about 11 cycles off from the true RUL.

        ### Dataset
        The model was trained on the renowned **NASA Turbofan Engine Degradation Simulation Dataset (FD001)**. This dataset contains simulated time-series data from 100 engines, each with multiple sensor readings, tracking their journey from a healthy state to failure.
        
        ### How to Use This App
        -   **Demo on Test Data:** Use this tab to see how the model performs on the official test data. You can select different engines and slide through their lifecycle to see the model's predictions change over time.
        -   **Predict on Your Data:** Use this tab to get a live prediction on your own data. You must upload a CSV file that has the same columns as the original data and at least 50 rows of data.
        """)

st.markdown("---")
st.write("Project developed as a portfolio piece for the KSA job market.")
