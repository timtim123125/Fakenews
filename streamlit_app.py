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
EnsemblePrediction = 0

# Extract features for prediction
def extract_features(text):
    return pd.DataFrame([{
        'clean_content': text,
        'text_len': len(text.split()),
        'punct_count': len(re.findall(r'[!?]', text)),
        'caps_count': sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    }])

# Prediction Function (Using the fine-tuned models)
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
            if input_type == "News Article":
                results.append(f"**{name}**: Prediction = {'🟥 Fake' if pred_adjusted == 0 else '🟩 Real'}" +
                               (f" (Fake Probability = `{prob:.2f}`)" if prob is not None else ""))
            else:
                results.append(f"**{name}**: Prediction = {'🟥 Phishing' if pred_adjusted == 0 else '🟩 Real'}" +
                                   (f" (Fake Probability = `{prob:.2f}`)" if prob is not None else ""))
        except Exception as e:
            results.append(f"**{name}**: ⚠️ Error: {e}")

    # Ensemble vote for final prediction
    majority_vote = Counter(weighted_votes).most_common(1)[0][0]
    final_prediction = '🟥 Fake' if majority_vote == 0 else '🟩 Real'
    EnsemblePrediction = 0 if majority_vote == 0 else 1
    results.append(f"\n**Ensemble**: Prediction = {final_prediction}")
    return "\n".join(results), final_prediction

# Run the inference with charts
import matplotlib.pyplot as plt
import streamlit as st

def run_infer_prediction(content_input):
    # Specify the model path for ONNX model (replace with actual model path)
    model_path = "qmodel.onnx"
    
    # Call the inference function from infer.py
    type_pred, queue_pred, type_probs, queue_probs = run_inference(content_input, model_path)
    
    # Check if type_probs and queue_probs are valid
    if not type_probs or not queue_probs:
        st.error("Error: The probabilities data is missing or invalid.")
        return

    # Generate a bar chart for the probabilities of types and queues
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot for Type Probabilities
    ax[0].bar([item['name'] for item in type_probs], [item['prob'] for item in type_probs])
    ax[0].set_title('Type Probabilities')
    ax[0].set_ylabel('Probability')
    ax[0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels if they are long

    # Plot for Queue Probabilities
    ax[1].bar([item['name'] for item in queue_probs], [item['prob'] for item in queue_probs])
    ax[1].set_title('Queue Probabilities')
    ax[1].set_ylabel('Probability')
    ax[1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels if they are long

    # Store the figure in session state so it persists across reruns
    st.session_state.chart = fig

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
            # Step 1: Use fine-tuned models to predict if it's fake or real
            result_string, fake_or_real_prediction = predict_fake_or_real(title_input + content_input)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"✅ Email received and classified for **Phishing Email**.\n\n**Title**: {title_input}\n\n{result_string}"
            })

            # Step 2: Run inference for additional results
            if EnsemblePrediction == 0:
                run_infer_prediction(title_input + ">>>" + content_input)
                # Include the chart in the assistant's message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Here are the inference results:",                    
                })

                # Display the chart after the response
                st.pyplot(st.session_state.chart)  

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
            result_string, fake_or_real_prediction = predict_fake_or_real(user_input)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"✅ News Article received and classified for **News Article**.\n\n{result_string}"
            })
            st.session_state.form_submitted = False
            st.rerun()
