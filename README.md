# Predictive Maintenance for Industrial Machinery using LSTM

LSTM that reads the last 30 cycles of sensor data and predicts (1) Remaining Useful Life (RUL)
and (2) the probability of failure within 30 cycles. Includes a Streamlit dashboard.

## Quick start (VS Code terminal, inside this folder)
    python -m venv venv
    venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
    pip install -r requirements.txt
    python -m src.train               # trains + evaluates (a few minutes on a laptop)
    streamlit run dashboard/streamlit_app.py

## Using real NASA C-MAPSS data
Download the dataset (Kaggle "NASA C-MAPSS" or NASA Prognostics Data Repository) and copy
train_FD001.txt, test_FD001.txt and RUL_FD001.txt into data/raw/, then run `python -m src.train` again.
Without them, the project uses auto-generated SYNTHETIC data (clearly labelled in the dashboard).
