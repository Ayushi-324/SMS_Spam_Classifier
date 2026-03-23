import streamlit as st
import pickle
import string
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 0. BASE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
DATA_PATH = os.path.join(BASE_DIR, "spam.csv")

# --- 1. NLTK DOWNLOAD SAFEGUARD ---
for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource=="punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

ps = PorterStemmer()

# --- 2. PREPROCESSING FUNCTION ---
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    
    filtered = [ps.stem(w) for w in tokens if w.isalnum() and w not in stopwords.words("english")]
    return " ".join(filtered)

# --- 3. LOAD MODEL AND VECTORIZER ---
tfidf_ready = False
model_ready = False

try:
    # Load model
    if os.path.exists(MODEL_PATH):
        model = pickle.load(open(MODEL_PATH, "rb"))
        model_ready = True
    else:
        st.error("❌ model.pkl not found!")

    # Load vectorizer
    if os.path.exists(VECTORIZER_PATH):
        tfidf = pickle.load(open(VECTORIZER_PATH, "rb"))
        # Check if fitted
        if hasattr(tfidf, "idf_"):
            tfidf_ready = True
        else:
            st.warning("⚠️ Vectorizer loaded but not fitted!")
    else:
        st.error("❌ vectorizer.pkl not found!")
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")

# --- 4. STREAMLIT UI ---
st.title("SMS / Email Spam Classifier")

input_sms = st.text_area("Enter your message:")

if st.button("Predict"):
    if not tfidf_ready or not model_ready:
        st.error("Model or vectorizer not ready. Ensure model.pkl, vectorizer.pkl, and spam.csv exist and are correct.")
    else:
        try:
            # Transform & predict
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            if result == 1:
                st.header("🚨 SPAM")
            else:
                st.header("✅ NOT SPAM")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# --- 5. Sidebar Debug Info ---
if tfidf_ready and model_ready:
    st.sidebar.success("Model Status: Ready ✅")
else:
    st.sidebar.warning("Model Status: Not Ready ⚠️")