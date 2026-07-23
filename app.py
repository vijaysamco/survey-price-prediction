import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Price Range Predictor",
    page_icon="🏷️",
    layout="wide"
)

# App Title & Header
st.title("🏷️ Consumer Survey Price Range Prediction App")
st.markdown("Predict a consumer's preferred **Price Range** based on demographic and behavioral survey responses using the top-performing **XGBoost** model.")

# --- 1. Load Data and Train Best Model (Cached for performance) ---
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('data\\engineered_survey_results.csv')
    
    X = df.drop(columns=['respondent_id', 'price_range']).copy()
    y = df['price_range'].copy()
    
    # Label encode target
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    
    # Label encode categorical/ordinal features
    label_cols = [
        'age_group', 
        'income_levels', 
        'health_concerns', 
        'consume_frequency(weekly)', 
        'preferable_consumption_size'
    ]
    
    X_encoded = X.copy()
    label_encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le
        
    # One-hot encode remaining categorical columns
    categorical_cols = X_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    X_final = pd.get_dummies(X_encoded, columns=categorical_cols)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_encoded, test_size=0.25, random_state=42
    )
    
    # Train Best Model: XGBoost
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42, 
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    
    feature_columns = X_final.columns.tolist()
    
    return model, le_y, label_encoders, feature_columns

model, le_y, label_encoders, feature_columns = load_and_train_model()

# --- 2. Streamlit Sidebar / Input Form ---
st.sidebar.header("📋 Input Consumer Survey Details")

gender = st.sidebar.selectbox("Gender", ["M", "F"])
zone = st.sidebar.selectbox("Geographic Zone", ["Rural", "Semi-Urban", "Urban", "Metro"])
occupation = st.sidebar.selectbox("Occupation", ["Student", "Working Professional", "Entrepreneur", "Retired"])
income_levels = st.sidebar.selectbox("Income Level", ["Not Reported", "<10L", "10L - 15L", "16L - 25L", "26L - 35L", "> 35L"])
age_group = st.sidebar.selectbox("Age Group", ["18-25", "26-35", "36-45", "46-55", "56-70"])

st.sidebar.subheader("Consumption Habits")
consume_freq = st.sidebar.selectbox("Weekly Consumption Frequency", ["0-2 times", "3-4 times", "5-7 times"])
current_brand = st.sidebar.selectbox("Current Brand Used", ["Established", "Newcomer"])
preferable_size = st.sidebar.selectbox("Preferable Size", ["Small (250 ml)", "Medium (500 ml)", "Large (1 L)"])
awareness_brands = st.sidebar.selectbox("Awareness of Other Brands", ["0 to 1", "2 to 4", "above 4"])
reasons_choosing = st.sidebar.selectbox("Reason for Choosing Brand", ["Price", "Quality", "Availability", "Brand Reputation"])
flavor_preference = st.sidebar.selectbox("Flavor Preference", ["Traditional", "Exotic"])
purchase_channel = st.sidebar.selectbox("Purchase Channel", ["Online", "Retail Store"])
packaging_pref = st.sidebar.selectbox("Packaging Preference", ["Simple", "Premium", "Eco-Friendly"])
health_concerns = st.sidebar.selectbox("Health Concerns Level", ["Low (Not very concerned)", "Medium (Moderately health-conscious)", "High (Very health-conscious)"])
typical_situations = st.sidebar.selectbox("Typical Consumption Situation", ["Active (eg. Sports, gym)", "Social (eg. Parties)", "Casual"])

# --- 3. Compute Derived Engineered Features ---
# Frequency Score & Awareness Score
freq_map = {'0-2 times': 1, '3-4 times': 2, '5-7 times': 3}
awareness_map = {'0 to 1': 1, '2 to 4': 2, 'above 4': 3}
f_score = freq_map[consume_freq]
a_score = awareness_map[awareness_brands]
cf_ab_score = round(f_score / (a_score + f_score), 2)

# Zone Score & Income Score (ZAS)
zone_map = {'Rural': 1, 'Semi-Urban': 2, 'Urban': 3, 'Metro': 4}
income_map = {'Not Reported': 0, '<10L': 1, '10L - 15L': 2, '16L - 25L': 3, '26L - 35L': 4, '> 35L': 5}
z_score = zone_map[zone]
i_score = income_map[income_levels]
zas_score = z_score * i_score

# Brand Switching Indicator (BSI)
bsi = 1 if (current_brand != "Established" and reasons_choosing in ["Price", "Quality"]) else 0

# --- 4. Main Page Display ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💡 Calculated Feature Summary")
    f_df = pd.DataFrame({
        "Engineered Feature": ["cf_ab_score (Consumption/Awareness Score)", "zas_score (Zone Affluence Score)", "bsi (Brand Switching Indicator)"],
        "Value": [cf_ab_score, zas_score, bsi]
    })
    st.table(f_df)

with col2:
    st.subheader("⚙️ Model Info")
    st.info("**Model Used:** XGBoost Classifier\n\n**Test Accuracy:** 92.28%\n\n**Target:** Price Range")

# --- 5. Make Prediction ---
if st.button("🚀 Predict Price Range", use_container_width=True):
    # Construct raw row dict
    raw_input = {
        'gender': gender,
        'zone': zone,
        'occupation': occupation,
        'income_levels': income_levels,
        'consume_frequency(weekly)': consume_freq,
        'current_brand': current_brand,
        'preferable_consumption_size': preferable_size,
        'awareness_of_other_brands': awareness_brands,
        'reasons_for_choosing_brands': reasons_choosing,
        'flavor_preference': flavor_preference,
        'purchase_channel': purchase_channel,
        'packaging_preference': packaging_pref,
        'health_concerns': health_concerns,
        'typical_consumption_situations': typical_situations,
        'age_group': age_group,
        'cf_ab_score': cf_ab_score,
        'zas_score': zas_score,
        'bsi': bsi
    }
    
    input_df = pd.DataFrame([raw_input])
    
    # Apply Label Encoding
    label_cols = ['age_group', 'income_levels', 'health_concerns', 'consume_frequency(weekly)', 'preferable_consumption_size']
    for col in label_cols:
        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
        
    # Apply One-Hot Encoding
    categorical_cols = input_df.select_dtypes(include=['object', 'category']).columns.tolist()
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Reindex to match training feature columns
    input_final = input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # Predict
    pred_class = model.predict(input_final)[0]
    pred_label = le_y.inverse_transform([pred_class])[0]
    pred_probs = model.predict_proba(input_final)[0]
    
    st.success(f"### 🎉 Predicted Preferred Price Range: **{pred_label}**")
    
    # Probability Breakdown Chart
    prob_df = pd.DataFrame({
        'Price Range': le_y.classes_,
        'Probability (%)': (pred_probs * 100).round(2)
    })
    st.subheader("📊 Class Confidence Breakdown")
    st.bar_chart(prob_df.set_index('Price Range'))