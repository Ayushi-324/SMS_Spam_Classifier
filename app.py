import streamlit as st
import pickle
import os
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# NLTK download safeguards
# -----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()

# -----------------------------
# Preprocessing function
# -----------------------------
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum() and t not in stopwords.words('english')]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)

# -----------------------------
# Load model & vectorizer safely
# -----------------------------
model_ready = False
vectorizer_ready = False

if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
    # Load model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    model_ready = True

    # Load vectorizer
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    # Check if vectorizer is fitted
    if hasattr(tfidf, "idf_"):
        vectorizer_ready = True

# Streamlit UI
st.title("SMS/Email Spam Classifier")

if model_ready and vectorizer_ready:
    st.sidebar.success("✅ Model & Vectorizer Ready")
    input_sms = st.text_area("Enter the message:")
    
    if st.button("Predict"):
        transformed = transform_text(input_sms)
        vector_input = tfidf.transform([transformed])
        prediction = model.predict(vector_input)[0]
        st.header("🚨 SPAM" if prediction else "✅ NOT SPAM")
else:
    st.sidebar.error("⚠️ Model or Vectorizer not ready.")
    st.write("Please ensure `model.pkl`, `vectorizer.pkl`, and `spam.csv` exist in your repo.")