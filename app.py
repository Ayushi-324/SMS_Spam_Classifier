
import streamlit as st
import pickle
import string
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
import nltk

# --- 0. SET UP PATHS ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
CSV_PATH = os.path.join(BASE_DIR, "spam.csv")

# --- 1. NLTK DOWNLOAD SAFEGUARDS ---
for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource=="punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

ps = PorterStemmer()

# --- 2. TEXT PREPROCESSING ---
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)

    # remove non-alphanumeric
    tokens = [t for t in tokens if t.isalnum()]

    # remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words("english") and t not in string.punctuation]

    # stemming
    tokens = [ps.stem(t) for t in tokens]

    return " ".join(tokens)

# --- 3. LOAD MODEL AND VECTORIZER ---
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()

try:
    tfidf = pickle.load(open(VECT_PATH, "rb"))
except Exception as e:
    st.warning(f"Vectorizer not loaded, will try to fit locally: {e}")
    tfidf = None

# --- 4. FIT TF-IDF LOCALLY IF NEEDED ---
if tfidf is None or not hasattr(tfidf, "idf_"):
    if os.path.exists(CSV_PATH):
        st.info("Vectorizer not fitted. Fitting using local spam.csv...")
        df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")
        df = df[['v1','v2']]
        df.columns = ['label','message']
        tfidf = TfidfVectorizer(max_features=3000)
        tfidf.fit(df['message'].apply(transform_text))

        # save back
        with open(VECT_PATH, "wb") as f:
            pickle.dump(tfidf, f)
        st.success("Vectorizer fitted and saved!")
    else:
        st.error(f"spam.csv not found at {CSV_PATH}. Please include it in repo.")
        st.stop()

# --- 5. STREAMLIT UI ---
st.title("SMS / Email Spam Classifier")

input_sms = st.text_area("Enter your message here:")

if st.button("Predict"):
    try:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.header("🚨 SPAM")
        else:
            st.header("✅ NOT SPAM")
    except NotFittedError:
        st.error("Vectorizer is still not fitted. Please check spam.csv or tfidf.pkl.")

# --- 6. DEBUG INFO ---
if hasattr(tfidf, "idf_"):
    st.sidebar.success("Model Status: Ready to go!")
else:
    st.sidebar.error("Model Status: Not Fitted")
