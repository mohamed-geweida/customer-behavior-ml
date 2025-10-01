# Customer Behavior – Spending Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white&style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white&style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white&style=for-the-badge)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white&style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?logo=numpy&logoColor=white&style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white&style=for-the-badge)
![Status](https://img.shields.io/badge/Status-DEMO-green?style=for-the-badge)
![Purpose](https://img.shields.io/badge/Purpose-Educational%20Project-blueviolet?style=for-the-badge)

A concise, end‑to‑end example that predicts a customer’s expected spending using a linear regression model with feature scaling, exposed via a simple Streamlit UI.

## Features

- Predicts spending from customer features (`feature_0` … `feature_4`).
- Scales inputs with `MinMaxScaler` and fits a `LinearRegression` model.
- Interactive Streamlit app for quick, manual what‑if analysis.
- Self‑contained sample dataset for immediate experimentation.

## Tech stack

- Python, NumPy, pandas
- scikit‑learn (MinMaxScaler, LinearRegression)
- Streamlit for the UI

## Quickstart

1. Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn
```

2. Run the app:

```bash
streamlit run app.py
```

3. In the browser UI, enter values for `feature_0` to `feature_4` and click "Predict Spending".

## Project structure

- `app.py` — Streamlit app for interactive predictions.
- `customer_behavior.ipynb` — Notebook for exploration and experiments.
- `customer_behavior_unsupervised.csv` — Sample dataset of customer features.

## How it works

- The project synthesizes a target variable `spending` from input features plus noise, then shifts values to ensure positivity.
- Features are scaled using `MinMaxScaler`.
- A `LinearRegression` model is trained on the scaled features to predict `spending`.
- During inference, user inputs are transformed with the same scaler before prediction.

## Notes

- Educational demo; not optimized for production or model governance.
- To try alternative models or feature engineering, start with `customer_behavior.ipynb`.
