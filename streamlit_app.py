import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Setup
st.set_page_config(page_title="AI News Checker Assistant", page_icon="🧠")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# App Title
st.title("I'm Veritas. Nice to meet you! 🧠")
st.caption("I can help you check whether a news passage is real or fake.")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Please enter a piece of text you'd like me to analyze for fake news."}]
if "awaiting_text" not in st.session_state:
    st.session_state.awaiting_text = True
if "saved_text" not in st.session_state:
    st.session_state.saved_text = ""

# Dummy models (for illustration)
fine_tuned_models = {
    "Logistic Regression": joblib.load("fine_tuned_logistic_regression.pkl"),
    "Naive Bayes": joblib.load("fine_tuned_naive_bayes.pkl"),
    "SVM (Linear)": joblib.load("fine_tuned_svm_(linear).pkl"),
    "Random Forest": joblib.load("fine_tuned_random_forest.pkl"),
    "XGBoost": joblib.load("fine_tuned_xgboost.pkl")
}

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

def extract_features(text):
    return pd.DataFrame([{
        'clean_content': clean_text(text),
        'text_len': len(text),
        'punct_count': sum([1 for c in text if c in string.punctuation]),
        'caps_count': sum([1 for c in text if c.isupper()])
    }])

# Adjusted model predictions for weighted voting
def predict_all_models(content_input):
    X = extract_features(content_input)
    results = []
    
    # Weights for each model
    model_weights = {
        "Logistic Regression": 3,  # Stronger influence
        "Naive Bayes": 1,          # Default influence
        "SVM (Linear)": 2,         # Medium influence
        "Random Forest": 1,        # Default influence
        "XGBoost": 3               # Stronger influence
    }

    # Predict with each model and store predictions and weights
        model_preds = []
        for name, model in fine_tuned_models.items():
            try:
                pred = model.predict(X)[0]
                weight = model_weights[name]
                model_preds.append((pred, weight))
    
                # Fix: Safe probability extraction
                prob = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    try:
                        if proba.shape[1] == 2:
                            prob = float(proba[0][1])
                        else:
                            prob = float(proba[0][0])
                        # Clamp to [0, 1]
                        if prob < 0 or prob > 1:
                            prob = min(max(prob, 0.0), 1.0)
                    except:
                        prob = None
                    
            results.append(f"{name}: Prediction = {'🟥 Fake' if pred else '🟩 Real'}")
            if prob is not None:
                results.append(f"  (Fake Probability = {prob:.2f})")
        except Exception as e:
            results.append(f"{name}: ⚠️ Model failed to load or predict: {e}")
            model_preds.append((None, 0))  # Append None with no weight for failed models

    # Weighted majority vote for ensemble prediction
    weighted_vote_real = sum(weight for pred, weight in model_preds if pred == 0)
    weighted_vote_fake = sum(weight for pred, weight in model_preds if pred == 1)
    
    # Choose the class with higher weighted sum
    if weighted_vote_fake > weighted_vote_real:
        ensemble_pred = 1  # Fake
        results.append(f"\n**Ensemble**: Prediction = 🟥 Fake")
    else:
        ensemble_pred = 0  # Real
        results.append(f"\n**Ensemble**: Prediction = 🟩 Real")

    return "\n".join(results)

# Chat message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input handling
if user_input := st.chat_input("Please enter your message"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.awaiting_text:
        st.session_state.saved_text = user_input
        st.session_state.awaiting_text = False

        # Get prediction from all models and the weighted ensemble
        result_string = predict_all_models(user_input)

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"✅ Text received.\n\n{result_string}"
        })

        # Reset to await new text again
        st.session_state.awaiting_text = True

    st.rerun()
