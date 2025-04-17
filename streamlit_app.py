import streamlit as st
import pandas as pd
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from collections import Counter
from infer import run_inference  # Import the function from infer.py
import matplotlib.pyplot as plt
import io

# Setup
st.set_page_config(page_title="Veritas - AI News & Email Checker", page_icon="🧠")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Title
st.title("I'm Veritas. Nice to meet you! 🧠")
st.caption("I can help you check whether a **news passage** or **email** is real or fake/phishing.")

# Sidebar
input_type = st.sidebar.selectbox("🗂️ Choose the input type:", ["News Article", "Phishing Email"])
expected_fake_label = 0 if input_type == "News Article" else 1

# Session State Init
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Please enter a text you'd like me to check."}]
if "awaiting_text" not in st.session_state:
    st.session_state.awaiting_text = True
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "chart_image" not in st.session_state:
    st.session_state.chart_image = None
if "show_chart" not in st.session_state:
    st.session_state.show_chart = False

# Load Models
model_files = {
    "Logistic Regression": "fine_tuned_logistic_regression.pkl",
    "Naive Bayes": "fine_tuned_naive_bayes.pkl",
    "SVM (Linear)": "fine_tuned_svm_(linear).pkl",
    "Random Forest": "fine_tuned_random_forest.pkl",
    "XGBoost": "fine_tuned_xgboost.pkl"
}
fine_tuned_models = {name: joblib.load(path) for name, path in model_files.items()}

# Extract features
def extract_features(text):
    return pd.DataFrame([{
        'clean_content': text,
        'text_len': len(text.split()),
        'punct_count': len(re.findall(r'[!?]', text)),
        'caps_count': sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    }])

# Predict function
def predict_fake_or_real(content_input):
    input_df = extract_features(content_input)
    results = []
    weighted_votes = []

    model_weights = {
        "Logistic Regression": 2,
        "Naive Bayes": 1,
        "SVM (Linear)": 2,
        "Random Forest": 1,
        "XGBoost": 3
    }

    for name, model in fine_tuned_models.items():
        try:
            pred = model.predict(input_df)[0]
            pred_adjusted = pred if expected_fake_label == 0 else 1 - pred
            weighted_votes.extend([pred_adjusted] * model_weights[name])

            prob = None
            if hasattr(model.named_steps['clf'], 'predict_proba'):
                prob = model.predict_proba(input_df)[0][1]
                prob = min(max(prob, 0.0), 1.0)
            label = '🟥 Fake' if pred_adjusted == 0 else '🟩 Real'
            phishing_label = '🟥 Phishing' if pred_adjusted == 0 else '🟩 Real'
            label_to_use = label if input_type == "News Article" else phishing_label
            results.append(f"**{name}**: Prediction = {label_to_use}" +
                           (f" (Fake Probability = `{prob:.2f}`)" if prob is not None else ""))
        except Exception as e:
            results.append(f"**{name}**: ⚠️ Error: {e}")

    majority_vote = Counter(weighted_votes).most_common(1)[0][0]
    final_prediction = '🟥 Fake' if majority_vote == 0 else '🟩 Real'
    ensemble_value = 0 if majority_vote == 0 else 1
    results.append(f"\n**Ensemble**: Prediction = {final_prediction}")
    return "\n".join(results), final_prediction, ensemble_value

# Chart generation function
def run_infer_prediction(content_input):
    model_path = "qmodel.onnx"
    type_pred, queue_pred, type_probs, queue_probs = run_inference(content_input, model_path)

    if not type_probs or not queue_probs:
        st.error("Error: The probabilities data is missing or invalid.")
        return None

    fig, ax = plt.subplots(1, 2, figsize=(18, 10))

    ax[0].bar([item['name'] for item in type_probs], [item['prob'] for item in type_probs])
    ax[0].set_title('Type Probabilities')
    ax[0].set_ylabel('Probability')
    ax[0].tick_params(axis='x', rotation=45)

    ax[1].bar([item['name'] for item in queue_probs], [item['prob'] for item in queue_probs])
    ax[1].set_title('Queue Probabilities')
    ax[1].set_ylabel('Probability')
    ax[1].tick_params(axis='x', rotation=45)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Show messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Phishing Email Input
if input_type == "Phishing Email":
    with st.form(key='email_form'):
        title_input = st.text_input("📧 Enter the title of the phishing email:")
        content_input = st.text_area("📝 Enter the content of the phishing email:")
        submit_button = st.form_submit_button("Check Email")

        if submit_button:
            st.session_state.form_submitted = True
            full_text = title_input + "\n\n" + content_input
            st.session_state.messages.append({"role": "user", "content": f"Title: {title_input}\n\nContent: {content_input}"})

            result_string, prediction, ensemble = predict_fake_or_real(full_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"✅ Email received and classified for **Phishing Email**.\n\n**Title**: {title_input}\n\n{result_string}"
            })

            if ensemble == 0:
                chart_buf = run_infer_prediction(title_input + ">>>" + content_input)
                if chart_buf:
                    st.session_state.chart_image = chart_buf
                    st.session_state.show_chart = True
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Here are the inference results:"
                    })

            st.session_state.form_submitted = False
            st.rerun()

# News Article Input
else:
    with st.form(key='news_form'):
        user_input = st.text_area("📄 Enter the news article content:")
        submit_button = st.form_submit_button("Check News Article")

        if submit_button:
            st.session_state.form_submitted = True
            st.session_state.messages.append({"role": "user", "content": user_input})


        if st.session_state.form_submitted and user_input:
            result_string, fake_or_real_prediction, _ = predict_fake_or_real(user_input)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"✅ News Article received and classified for **News Article**.\n\n{result_string}"
            })
            st.session_state.form_submitted = False
            st.rerun()
# Display messages + chart if applicable
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Show chart immediately after the assistant's response
        if message["role"] == "assistant" and "inference results" in message["content"].lower():
            if st.session_state.get("chart_image"):
                st.image(st.session_state.chart_image, use_column_width=True)

