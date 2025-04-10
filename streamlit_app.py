import streamlit as st
import pandas as pd
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords

# Setup
st.set_page_config(page_title="AI News Checker Assistant", page_icon="🧠")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# App Title
st.title("I'm Veritas. Nice to meet you! 🧠")
st.caption("I can help you check whether a news passage or an email is real or fake/phising")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Please enter a piece of text you'd like me to analyze for fake news."}]
if "awaiting_text" not in st.session_state:
    st.session_state.awaiting_text = True
if "saved_text" not in st.session_state:
    st.session_state.saved_text = ""
    
# Clean content function (example)
def clean_content(text):
    # Basic cleaning steps: remove punctuation, convert to lowercase, etc.
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load all trained pipelines
fine_tuned_models = {
    "Logistic Regression": joblib.load("fine_tuned_logistic_regression.pkl"),
    "Naive Bayes": joblib.load("fine_tuned_naive_bayes.pkl"),
    "SVM (Linear)": joblib.load("fine_tuned_svm_(linear).pkl"),
    "Random Forest": joblib.load("fine_tuned_random_forest.pkl"),
    "XGBoost": joblib.load("fine_tuned_xgboost.pkl")
}

# Predict using all models
def predict_all_models(content_input):
    results = []

    model_weights = {
        "Logistic Regression": 3,
        "Naive Bayes": 1,
        "SVM (Linear)": 2,
        "Random Forest": 1,
        "XGBoost": 3
    }

    model_preds = []

    for name, model in fine_tuned_models.items():
        try:
            # Match the structure used during training
            input_df = pd.DataFrame([{
                'clean_content': clean_content(content_input),
                'text_len': len(content_input.split()),
                'punct_count': len(re.findall(r'[!?]', content_input)),
                'caps_count': sum(1 for w in content_input.split() if w.isupper() and len(w) > 1)
            }])

            pred = model.predict(input_df)[0]
            weight = model_weights[name]
            model_preds.append((pred, weight))

            prob = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)
                if proba.shape[1] == 2:
                    prob = float(proba[0][1])
                    prob = min(max(prob, 0.0), 1.0)
            results.append(f"{name}: Prediction = {'🟥 Fake' if pred == 1 else '🟩 Real'}")
            if prob is not None:
                results.append(f"  (Fake Probability = {prob:.2f})")

        except Exception as e:
            results.append(f"{name}: ⚠️ Model failed: {e}")
            model_preds.append((None, 0))

    weighted_vote_fake = sum(weight for pred, weight in model_preds if pred == 1)
    weighted_vote_real = sum(weight for pred, weight in model_preds if pred == 0)

    if weighted_vote_fake > weighted_vote_real:
        results.append(f"\n**Ensemble**: Prediction = 🟥 Fake")
    else:
        results.append(f"\n**Ensemble**: Prediction = 🟩 Real")

    return "\n".join(results)

# Chat history display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if user_input := st.chat_input("Please enter your message"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.awaiting_text:
        st.session_state.saved_text = user_input
        st.session_state.awaiting_text = False

        result_string = predict_all_models(user_input)

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"✅ Text received.\n\n{result_string}"
        })

        st.session_state.awaiting_text = True

    st.rerun()
