import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import matplotlib.pyplot as plt

# App Configuration
st.set_page_config(page_title="XAI for Network Anomaly Detection", layout="wide")

# Helper Functions
@st.cache_data
def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: '{filepath}' not found. Please run `python get_data.py` first.")
        return None

def preprocess_data(df):
    df_processed = df.copy()
    # Drop columns that are identifiers or labels
    cols_to_drop = ['id', 'attack_cat', 'label', 'srcip', 'dstip']
    df_processed = df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns], errors='ignore')

    # Encode categorical features and scale numerical ones
    for col in df_processed.select_dtypes(include=['object', 'category']).columns:
        df_processed[col] = LabelEncoder().fit_transform(df_processed[col])
    
    numerical_cols = df_processed.select_dtypes(include=np.number).columns
    if not df_processed[numerical_cols].empty:
        df_processed[numerical_cols] = StandardScaler().fit_transform(df_processed[numerical_cols])
        
    return df_processed

# Main UI
st.title("🔬 Explainable AI for Network Anomaly Detection")

original_df = load_data("UNSW_NB15_training-set.csv")

if original_df is not None:
    st.header("1. Raw Network Data Preview")
    st.dataframe(original_df.head())

    processed_df = preprocess_data(original_df)

    st.header("2. Detect Anomalies")
    if st.button("Run Anomaly Detection"):
        with st.spinner("Training model and finding anomalies..."):
            model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
            model.fit(processed_df)
            predictions = model.predict(processed_df)
            original_df['anomaly'] = predictions
            
            st.session_state['model'] = model
            st.session_state['processed_df'] = processed_df
            st.session_state['anomalies'] = original_df[original_df['anomaly'] == -1]
            st.session_state['has_run'] = True

        st.success(f"Detection complete! Found {len(st.session_state['anomalies'])} potential anomalies.")

    if st.session_state.get('has_run', False):
        st.header("3. Explain a Detected Anomaly with SHAP")
        anomalies_df = st.session_state['anomalies']
        
        st.subheader("Detected Anomalies")
        st.dataframe(anomalies_df)
        
        selected_index = st.selectbox("Choose an anomaly to explain:", options=anomalies_df.index.tolist())
        
        if selected_index is not None:
            with st.spinner("Generating SHAP explanation..."):
                model = st.session_state['model']
                processed_df = st.session_state['processed_df']
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(processed_df.loc[selected_index])
                
                st.info("This plot shows features contributing to the anomaly score. Red features push the score higher (more anomalous), while blue features push it lower.")
                
                fig = shap.force_plot(explainer.expected_value, shap_values, processed_df.loc[selected_index], matplotlib=True, show=False)
                st.pyplot(fig, bbox_inches='tight', clear_figure=True)