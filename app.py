import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(page_title="XAI for Network Anomaly Detection", layout="wide")

# --- Helper Functions ---

@st.cache_data
def load_data(filepath):
    """Loads and caches data from a CSV file."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: '{filepath}' not found. Please run `python get_data.py` first.")
        return None

def preprocess_data(df):
    """Encodes categorical features and scales numerical features."""
    df_processed = df.copy()
    # Drop columns that are identifiers or labels (not features for unsupervised learning)
    cols_to_drop = ['id', 'attack_cat', 'label', 'srcip', 'dstip']
    df_processed = df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns], errors='ignore')

    # Encode categorical features
    for col in df_processed.select_dtypes(include=['object', 'category']).columns:
        df_processed[col] = LabelEncoder().fit_transform(df_processed[col])
    
    # Scale numerical features
    numerical_cols = df_processed.select_dtypes(include=np.number).columns
    if not df_processed[numerical_cols].empty:
        df_processed[numerical_cols] = StandardScaler().fit_transform(df_processed[numerical_cols])
        
    return df_processed

# --- Main Application UI and Logic ---

st.title("🔬 Explainable AI for Network Anomaly Detection")

# Load data
original_df = load_data("UNSW_NB15_training-set.csv")

if original_df is not None:
    st.header("1. Raw Network Data Preview")
    st.dataframe(original_df.head())

    # Preprocess data
    processed_df = preprocess_data(original_df)

    st.header("2. Detect Anomalies")
    if st.button("Run Anomaly Detection"):
        with st.spinner("Training model and finding anomalies..."):
            # Train the model
            model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
            model.fit(processed_df)
            
            # Get predictions and add to the original dataframe
            predictions = model.predict(processed_df)
            original_df['anomaly'] = predictions
            
            # Store results in session state
            st.session_state['model'] = model
            st.session_state['processed_df'] = processed_df
            st.session_state['anomalies'] = original_df[original_df['anomaly'] == -1]
            st.session_state['has_run'] = True

        st.success(f"Detection complete! Found {len(st.session_state['anomalies'])} potential anomalies.")

    # This block runs after the detection is complete
    if st.session_state.get('has_run', False):
        st.header("3. Explain a Detected Anomaly with SHAP")
        anomalies_df = st.session_state['anomalies']
        
        st.subheader("Detected Anomalies")
        st.dataframe(anomalies_df)
        
        # Allow user to select an anomaly to explain
        selected_index = st.selectbox("Choose an anomaly to explain:", options=anomalies_df.index.tolist())
        
        # --- CORRECTED PLOTTING LOGIC ---
        if selected_index is not None:
            with st.spinner("Generating SHAP explanation..."):
                model = st.session_state['model']
                processed_df = st.session_state['processed_df']
                explainer = shap.TreeExplainer(model)
                
                # Use the explainer directly on a single-row DataFrame
                # Note the double brackets [[...]] to ensure the input is 2D
                explanation = explainer(processed_df.loc[[selected_index]])
                
                st.info("This waterfall plot shows how each feature pushes the model's output from the base value to the final prediction. Red features increase the anomaly score, while blue features decrease it.")
                
                # Pass the first (and only) explanation from the object to the plot
                shap.waterfall_plot(explanation[0], max_display=15, show=False)
                
                fig = plt.gcf() # Get the current plot
                st.pyplot(fig, bbox_inches='tight', clear_figure=True)
