# 🔬 Explainable AI for Network Anomaly Detection

This project provides an interactive web application for detecting and explaining anomalies in network traffic data. It uses an **Isolation Forest** model for detection and **SHAP (SHapley Additive exPlanations)** to provide human-readable explanations for each detected anomaly.

## ✨ Key Features
- **Anomaly Detection**: Employs an unsupervised Isolation Forest algorithm to identify unusual network flows.
- **Explainability (XAI)**: Integrates SHAP to show which network features (e.g., port, packet size, duration) contributed most to an anomaly score.
- **Interactive UI**: Built with Streamlit for an easy-to-use web interface.
- **Reproducible**: Includes a requirements file and data download script for easy setup.

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
```

### 2. Create and Activate a Virtual Environment
- **Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
Install all required libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
Run the provided script to download the UNSW-NB15 sample dataset.
```bash
python get_data.py
```

### 5. Run the Streamlit App
Launch the application.
```bash
streamlit run app.py
```
Open your web browser and navigate to the local URL provided by Streamlit.

## 🤝 How to Contribute
We welcome contributions! Please follow these steps:
1.  **Fork** the repository.
2.  Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a **Pull Request**.

---
*Created by SumitGhughtyal*