import streamlit as st
import pandas as pd
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from collections import Counter
from infer import run_inference  # Import the function from infer.py

# Setup
st.set_page_config(page_title="Veritas - AI News & Email Checker", page_icon="🧠")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Title
st.title("I'm Veritas. Nice to meet you! 🧠")
st.caption("I can help you check whether a **news passage** or **email** is real or fake/phishing.")

# Sidebar: Input Type Switcher
input_type = st.sidebar.selectbox("🗂️ Choose the input type:", ["News Article", "Phishing Email"])
expected_fake_label = 0 if input_type == "News Article" else 1  # Invert logic if it's Enron (where 1 = Fake)

# Session State Init
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Please enter a text you'd like me to check."}]
if "awaiting_text" not in st.session_state:
    st.session_state.awaiting_text = True
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

# Load Models (if still needed for other purposes)
model_files = {
    "Logistic Regression": "fine_tuned_logistic_regression.pkl",
    "Naive Bayes": "fine_tuned_naive_bayes.pkl",
    "SVM (Linear)": "fine_tuned_svm_(linear).pkl",
    "Random Forest": "fine_tuned_random_forest.pkl",
    "XGBoost": "fine_tuned_xgboost.pkl"
}
fine_tuned_models = {name: joblib.load(path) for name, path in model_files.items()}

# Prediction Function (Using infer.py)
def run_infer_prediction(text):
    # Specify the model path for ONNX model (replace with actual model path)
    model_path = "Deeplearning with Reflect/assets/qmodel.onnx"
    
    # Call the inference function from infer.py
    type_pred, queue_pred, type_probs, queue_probs = run_inference(text, model_path)
    
    # Return the results formatted
    result_string = f"Type Prediction: {type_pred}\nQueue Prediction: {queue_pred}\n"
    
    result_string += "\nType Probabilities:\n"
    for item in type_probs:
        result_string += f"{item['name']}: {item['prob']}\n"
        
    result_string += "\nQueue Probabilities:\n"
    for item in queue_probs:
        result_string += f"{item['name']}: {item['prob']}\n"
        
    return result_string

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input for phishing email (separate title and content)
if input_type == "Phishing Email":
    with st.form(key='email_form'):
        title_input = st.text_input("📧 Enter the title of the phishing email:")
        content_input = st.text_area("📝 Enter the content of the phishing email:")

        submit_button = st.form_submit_button("Check Email")

        if submit_button:
            st.session_state.form_submitted = True
            st.session_state.messages.append({"role": "user", "content": f"Title: {title_input}\n\nContent: {content_input}"})

        if st.session_state.form_submitted and title_input and content_input:
            # Prediction logic using the inference function
            result_string = run_infer_prediction(content_input)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"✅ Email received and classified for **Phishing Email**.\n\n**Title**: {title_input}\n\n{result_string}"
            })
            st.session_state.form_submitted = False
            st.rerun()

else:
    # Handle News Article input (using form and rerun)
    with st.form(key='news_form'):
        user_input = st.text_area("📄 Enter the news article content:")

        submit_button = st.form_submit_button("Check News Article")

        if submit_button:
            st.session_state.form_submitted = True
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Trigger rerun to update the page
            st.rerun()

        if st.session_state.form_submitted and user_input:
            # Prediction logic using the inference function
            result_string = run_infer_prediction(user_input)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"✅ News Article received and classified for **News Article**.\n\n{result_string}"
            })
            st.session_state.form_submitted = False
            st.rerun()
