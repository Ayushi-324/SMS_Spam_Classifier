import nltk
from nltk.data import find
import streamlit as st
import pickle
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Ensure NLTK resources ---
try:
    find('tokenizers/punkt')
except:
    nltk.download('punkt')
try:
    find('corpora/stopwords')
except:
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stopwords.words('english')]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# --- Load model & vectorizer ---
model_path = 'model.pkl'
vectorizer_path = 'vectorizer.pkl'

if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))
else:
    st.error("❌ Model not found!")

if os.path.exists(vectorizer_path):
    tfidf = pickle.load(open(vectorizer_path, 'rb'))
else:
    st.error("❌ Vectorizer not found!")

# --- Streamlit UI ---
st.title("SMS/Email Spam Classifier")
input_sms = st.text_area("Enter your message:")

if st.button("Predict"):
    if not hasattr(tfidf, 'idf_'):
        st.error("Vectorizer not ready. Ensure 'vectorizer.pkl' exists and is fitted.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        st.header("🚨 SPAM" if result == 1 else "✅ NOT SPAM")