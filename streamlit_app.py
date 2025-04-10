import streamlit as st
import joblib
import nltk
import re
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

# Setup
st.set_page_config(page_title="AI News Checker Assistant", page_icon="🧠")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# App Title
st.title("I'm Veritas. Nice to meet you! 🧠")
st.caption("I can help you check whether a news passage is real or fake.")
st.markdown("NEW Testing")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Please enter a piece of text you'd like me to analyze for fake news."}]
if "awaiting_text" not in st.session_state:
    st.session_state.awaiting_text = True
if "saved_text" not in st.session_state:
    st.session_state.saved_text = ""

# Load trained models
fine_tuned_models = {
    "Logistic Regression": joblib.load("fine_tuned_logistic_regression.pkl"),
    "Naive Bayes": joblib.load("fine_tuned_naive_bayes.pkl"),
    "SVM (Linear)": joblib.load("fine_tuned_svm_(linear).pkl"),
    "Random Forest": joblib.load("fine_tuned_random_forest.pkl"),
    "XGBoost": joblib.load("fine_tuned_xgboost.pkl")
}

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # normalize whitespace
    return text

# Prepare input function: cleans text and prepares features
def prepare_input(content_input):
    cleaned_text = clean_text(content_input)
    
    # Extract additional features (length, punctuation count, capital letter count)
    features = {
        'clean_content': cleaned_text,
        'text_len': len(cleaned_text.split()),
        'punct_count': len(re.findall(r'[!?]', cleaned_text)),
        'caps_count': sum(1 for w in cleaned_text.split() if w.isupper() and len(w) > 1)
    }

    # Create a DataFrame (with the required structure)
    input_df = pd.DataFrame([features])
    return input_df

# Prediction function
def predict_all_models(content_input):
    results = []
    input_df = prepare_input(content_input)  # Prepare the structured input

    # Feature transformer and model prediction
    feature_union = ColumnTransformer([
        ('tfidf', TfidfVectorizer(max_df=0.7, ngram_range=(1, 2), max_features=5000), 'clean_content'),
        ('num', MinMaxScaler(), ['text_len', 'punct_count', 'caps_count'])
    ])

    for name, model in fine_tuned_models.items():
        try:
            # Apply the ColumnTransformer to the input (this includes TF-IDF + scaling)
            transformed_input = feature_union.fit_transform(input_df)

            # Predict using the model
            pred = model.predict(transformed_input)[0]
            result_line = f"**{name}**: Prediction = {'🟥 Fake' if pred == 0 else '🟩 Real'}"

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(transformed_input)
                if proba.shape[1] == 2:
                    fake_prob = proba[0][0]  # Probability of class 0 (Fake)
                    result_line += f"  (Fake Probability = {fake_prob:.2f})"

            results.append(result_line)

        except Exception as e:
            results.append(f"**{name}**: ⚠️ Model failed: {e}")

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
