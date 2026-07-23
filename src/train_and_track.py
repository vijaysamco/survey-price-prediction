# MLflow and DagsHub integration script

import os
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. Initialize DagsHub MLflow Remote Tracking
# ==========================================
DAGSHUB_USERNAME = "vijaysamco"   # Replace with your DagsHub username
DAGSHUB_REPO = "survey-price-prediction"     # Replace with your repository name

# Automatically configure remote MLflow tracking URL and authentication credentials
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)

# Set the MLflow Experiment Name
mlflow.set_experiment("Price_Range_Prediction_Models")

# ==========================================
# 2. Data Preparation
# ==========================================
# df = pd.read_csv('engineered_survey_results.csv')

# Define features (X) and target (y)
X = df_final.drop(columns=['respondent_id', 'price_range']).copy()
y = df_final['price_range'].copy()

# Label encode target variable
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

# Label encoding for designated ordinal/categorical columns
label_cols = [
    'age_group', 
    'income_levels', 
    'health_concerns', 
    'consume_frequency(weekly)', 
    'preferable_consumption_size'
]

X_encoded = X.copy()
for col in label_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

# One-hot encoding for remaining categorical features
categorical_cols = X_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
X_final = pd.get_dummies(X_encoded, columns=categorical_cols)

# Train-test split (75% train / 25% test, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_encoded, test_size=0.25, random_state=42
)

# ==========================================
# 3. Define Models (Including XGBoost)
# ==========================================
models = {
    "Gaussian_Naive_Bayes": GaussianNB(),
    "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Support_Vector_Machine": SVC(random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42, 
        eval_metric='mlogloss'
    )
}

# ==========================================
# 4. Train, Track & Publish to DagsHub MLflow
# ==========================================
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log hyperparameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("test_size", 0.25)
        mlflow.log_param("random_state", 42)
        
        if model_name == "XGBoost":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("max_depth", 6)

        # Log evaluation metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log and register model artifact to DagsHub
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )
        
        print(f"✅ Logged '{model_name}' to DagsHub MLflow | Accuracy: {acc:.4f}")

print("\n🎉 All models (including XGBoost) logged and published to DagsHub!")