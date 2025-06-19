# Predictive Maintenance for Industrial Machinery: RUL Prediction

This project demonstrates an end-to-end machine learning workflow to predict the Remaining Useful Life (RUL) of industrial machinery, specifically using NASA's Turbofan Engine Degradation Simulation dataset.

This portfolio piece is designed to showcase skills in time-series analysis, deep learning (LSTMs), and iterative model development, all of which are highly relevant to Saudi Arabia's Vision 2030 initiative, particularly in the energy, manufacturing, and industrial sectors.

## Project Goal

The primary objective is to build a model that can accurately predict how many operational cycles an engine has left before it is likely to fail. This is a critical task in **predictive maintenance**, as it allows organizations to schedule repairs proactively, minimizing downtime and maximizing operational efficiency.

## Dataset

The project uses the **NASA Turbofan Engine Degradation Simulation Dataset (FD001)**. This is a classic dataset for prognostics and health management, containing multivariate time-series data from 100 turbofan engines under simulated real-world conditions. Each engine starts with a different degree of initial wear and manufacturing variation and runs until failure.

## Methodology

The project followed a structured, iterative development process:

1.  **Exploratory Data Analysis (EDA):** Initial analysis to understand the data's structure, identify key features, and find non-informative (constant) columns.
2.  **Data Preprocessing:** Cleaning the data, assigning meaningful column names, and removing the constant columns.
3.  **Feature Engineering:** Calculating the target variable, **Remaining Useful Life (RUL)**, for each time step of every engine's life.
4.  **Data Scaling:** Applying Min-Max scaling to all sensor and setting features to prepare them for the neural network.
5.  **Sequence Creation:** Transforming the time-series data into 3D sequences suitable for input into an LSTM model.
6.  **Iterative Model Development:**
    *   **Baseline Model:** A simple two-layer LSTM was built to establish an initial performance benchmark. This revealed that the model was making "safe," averaged predictions.
    *   **Improved Model (RUL Clipping):** To force the model to focus on more critical, near-failure predictions, the RUL values were clipped at a ceiling of 125 cycles. This is a common and effective technique in RUL prediction.
    *   **Final Model (Longer Training & Early Stopping):** Analysis showed the model was undertrained. The final iteration involved training for a much longer period (100 epochs) while using an `EarlyStopping` callback. This callback monitored validation loss and automatically stopped the training at the optimal point (after 37 epochs) to achieve the best performance without overfitting.

## Final Results

The final model achieved a **Root Mean Squared Error (RMSE) of 10.96** on the validation set. This indicates a very high level of accuracy, with the model's predictions being, on average, only ~11 cycles away from the true RUL.

The success of the model is clearly visualized in the final evaluation plot, which shows the model's predictions closely tracking the "Perfect Prediction" line.

![Final Evaluation Plot](final_evaluation_plot.png)

## How to Replicate

The project is organized into a series of numbered Python scripts that reflect the development workflow:

1.  `1_load_private_data.py`: Loads the raw data.
2.  `2_eda.py`: Performs initial exploratory analysis.
3.  `3_preprocess_data.py`: Cleans and preprocesses the data.
4.  `4_feature_engineering.py`: Calculates the RUL target variable.
5.  `5_scale_data.py`: Scales the feature data.
6.  `6_create_sequences.py`: Creates 3D sequences for the LSTM.
7.  `7_train_model.py`: Trains the initial baseline model.
8.  `8_evaluate_model.py`: Evaluates the baseline model and identifies issues.
9.  `9_train_with_clipped_rul.py`: Trains the intermediate model with RUL clipping.
10. `10_final_model_training.py`: Trains the final, high-performing model using RUL clipping and early stopping.

## Project Artifacts

*   `lstm_rul_predictor_FINAL.h5`: The saved, high-performance Keras model.
*   `final_evaluation_plot.png`: The plot showing the final model's excellent performance.
*   `CMAPSSData/`: Directory containing the raw dataset.
*   Numbered Python scripts (`.py` files) detailing the complete workflow.
