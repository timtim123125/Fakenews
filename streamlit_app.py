import streamlit as st
import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords

# Setup
st.set_page_config(page_title="AI News Checker Assistant", page_icon="🧠")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# App Title
st.title("I'm Veritas. Nice to meet you! 🧠")
st.caption("I can help you check whether a news passage is real or fake.")
st.markdown("🧠 Note: According to the WELFake dataset: 0 = Fake, 1 = Real.")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Please enter a piece of text you'd like me to analyze for fake news."}]
if "awaiting_text" not in st.session_state:
    st.session_state.awaiting_text = True
if "saved_text" not in st.session_state:
    st.session_state.saved_text = ""

# Load all trained pipelines
fine_tuned_models = {
    "Logistic Regression": joblib.load("fine_tuned_logistic_regression.pkl"),
    "Naive Bayes": joblib.load("fine_tuned_naive_bayes.pkl"),
    "SVM (Linear)": joblib.load("fine_tuned_svm_(linear).pkl"),
    "Random Forest": joblib.load("fine_tuned_random_forest.pkl"),
    "XGBoost": joblib.load("fine_tuned_xgboost.pkl")
}

# Clean content function (example)
def clean_content(text):
    # Basic cleaning steps: remove punctuation, convert to lowercase, etc.
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Predict using all models
def predict_all_models(content_input):
    results = []

    # Define weights for the models
    model_weights = {
        "Logistic Regression": 3,
        "Naive Bayes": 1,
        "SVM (Linear)": 2,
        "Random Forest": 1,
        "XGBoost": 3
    }

    model_preds = []

    # Clean the content before further processing
    cleaned_content = clean_content(content_input)

    # Prepare input features (directly from cleaned_content)
    text_len = len(cleaned_content.split())
    punct_count = len(re.findall(r'[!?]', cleaned_content))
    caps_count = sum(1 for word in cleaned_content.split() if word.isupper() and len(word) > 1)

    # Create a DataFrame with cleaned content and features
    input_df = pd.DataFrame([{
        'clean_content': cleaned_content,  # Directly using the cleaned content
        'text_len': text_len,
        'punct_count': punct_count,
        'caps_count': caps_count
    }])

    # Loop through all models to make predictions
    for name, model in fine_tuned_models.items():
        try:
            # Predict with the model
            pred = model.predict(input_df)[0]
            weight = model_weights[name]
            model_preds.append((pred, weight))

            # Calculate probability (if available)
            prob_fake = None
            prob_real = None

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)
                if proba.shape[1] == 2:  # Check if the model outputs a probability for two classes (Fake, Real)
                    prob_fake = proba[0][0]  # Probability for Fake (class 0)
                    prob_real = proba[0][1]  # Probability for Real (class 1)
                    prob_fake = min(max(prob_fake, 0.0), 1.0)  # Clip probability between 0 and 1
                    prob_real = min(max(prob_real, 0.0), 1.0)  # Clip probability between 0 and 1

            # Collect results with probabilities
            if pred == 0:
                results.append(f"{name}: Prediction = 🟥 Fake (Fake Probability = {prob_fake:.2f})")
            else:
                results.append(f"{name}: Prediction = 🟩 Real (Real Probability = {prob_real:.2f})")

        except Exception as e:
            results.append(f"{name}: ⚠️ Model failed: {e}")
            model_preds.append((None, 0))


# Calculate the weighted ensemble prediction based on model outputs
    weighted_vote_fake = sum(weight for pred, weight in model_preds if pred == 0)
    weighted_vote_real = sum(weight for pred, weight in model_preds if pred == 1)

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
