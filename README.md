# Breast Cancer Classifier

A machine learning web app that predicts whether a breast tumor is **benign** or **malignant** from cell nuclei measurements, built on the Wisconsin Breast Cancer dataset and deployed as an interactive [Streamlit](https://streamlit.io/) app.

## Overview

The app takes 10 cell nuclei measurements as input and returns a prediction with a confidence score. It's built as a full, working pipeline i.e data cleaning, feature selection, model training/evaluation, and deployement.

The model trains directly on the 10 most important features instead of all 30 features using the **random forest algorithm**.

## Dataset

- **Source:** [Wisconsin Breast Cancer Diagnostic dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) (`data.csv`)
- **569 samples**, 30 numeric features describing cell nuclei from digitized images of breast mass biopsies (radius,- texture, perimeter, area, smoothness, concavity, etc.), each reported as a mean, standard error, and "worst" value
- **Target:** `diagnosis` — Malignant (M) or Benign (B), label-encoded to 1/0
- Class balance: 357 benign, 212 malignant
- No missing values or duplicates in the feature set (an empty `Unnamed: 32` column and the `id` column were dropped)

## Approach

1. **Data cleaning** — dropped the unused `id` and empty `Unnamed: 32` columns, encoded the target with `LabelEncoder`.
2. **Train/test split** — 80/20 split (`random_state=42`) for honest evaluation.
3. **Feature scaling** — standardized features with `StandardScaler` (zero mean, unit variance), since Logistic Regression is scale-sensitive.
4. **Feature selection with Random Forest** — rather than exposing a user to 30 input fields, a `RandomForestClassifier` (300 trees) was fit on the full feature set purely to rank features by importance (`feature_importances_`). The **top 10 most predictive features** were kept:

   `perimeter_worst`, `area_worst`, `concave points_worst`, `concave points_mean`, `radius_worst`, `perimeter_mean`, `concavity_mean`, `radius_mean`, `area_mean`, `concavity_worst`

5. **Final model** — a `LogisticRegression` was trained on just these 10 scaled features (its own train/test split and `StandardScaler`, fit independently of the full-feature scaler above), reaching **97.4% test accuracy** with a much smaller input surface for the app.
6. **Deployment prep** — the trained model, scaler, selected feature names, and per-feature "usual range" statistics (for the UI hints) were serialized with `joblib` for the app to load without retraining.

This is a deliberate two-model design: **Random Forest as a feature-importance ranker**, **Logistic Regression as the final classifier**, each used for what it's good at.

## App (`app.py`)

Built with Streamlit:

- Renders one input per selected feature, labeled in plain language (e.g. "Largest Concave Points" instead of `concave points_worst`), with the dataset's typical range shown as a guide (users can still enter values outside it).
- On clicking **Predict**, the input vector is scaled with the saved `StandardScaler` and passed to the saved `LogisticRegression` model.
- Displays the predicted class (Benign  / Malignant ) along with the model's confidence (`predict_proba`).

## Project Structure

```
BreastCancerProj/
├── app.py                                # Streamlit app
├── Breast_cancer_detection_model.ipynb   # Data prep, training, evaluation, feature selection
├── data.csv                              # Wisconsin Breast Cancer dataset
├── model.pkl                             # Trained Logistic Regression model (10 features)
├── scaler.pkl                            # Fitted StandardScaler
├── feature_names.pkl                     # Selected top-10 feature names
├── feature_stats.pkl                     # Per-feature UI ranges/defaults
└── requirements.txt
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The model artifacts (`.pkl` files) are already included, so the app runs immediately without retraining. To retrain, run through `Breast_cancer_detection_model.ipynb` end to end — it regenerates all four `.pkl` files.

## Tech Stack

- **Python**, **pandas** / **NumPy** — data handling
- **scikit-learn** — `LogisticRegression`, `RandomForestClassifier`, `StandardScaler`, `train_test_split`, `LabelEncoder`
- **Streamlit** — interactive web UI
- **joblib** — model serialization

## Results

| Model | Features used | Test Accuracy |
|---|---|---|
| Logistic Regression | Top 10 (Random Forest–ranked) | 97.4% |

