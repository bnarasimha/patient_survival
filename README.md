# Patient Survival Prediction

A machine learning application that predicts survival outcomes for patients with heart failure based on their clinical records. The project uses XGBoost for classification and Gradio for an interactive web interface.

## Features

- **XGBoost Classification Model** trained on heart failure clinical data
- **Outlier Handling** using IQR-based whisker capping for robust predictions
- **Interactive Web UI** built with Gradio for real-time predictions
- **Dockerized Deployment** for easy containerized hosting
- **CI/CD Pipeline** via GitHub Actions for automated training, Docker build, and push

## Tech Stack

- **Language:** Python 3.9+
- **ML Framework:** XGBoost, scikit-learn
- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib
- **Web Interface:** Gradio
- **Containerization:** Docker
- **CI/CD:** GitHub Actions

## Setup / Installation

### Local

```bash
git clone https://github.com/bnarasimha21/patient-survival.git
cd patient-survival
pip install -r requirements.txt
```

### Docker

```bash
docker build -t patient-survival .
docker run -p 8001:8001 patient-survival
```

## Usage

### 1. Train the Model

```bash
python train.py
```

This trains an XGBoost classifier on the heart failure clinical records dataset and saves the model as `xgboost-model.pkl`.

### 2. Run the Web Application

```bash
python app.py
```

Open your browser at `http://localhost:8001`. Enter patient clinical parameters (age, anaemia, creatinine phosphokinase, diabetes, ejection fraction, high blood pressure, platelets, serum creatinine, serum sodium, sex, smoking status, and follow-up time) to get a survival prediction.

## Dataset

The project uses the [Heart Failure Clinical Records Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) containing 299 patient records with 12 clinical features and a binary death event target.

## License

MIT
