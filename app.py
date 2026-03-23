import streamlit as st
import pickle
import string
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk

# --- 0. NLTK Download Safeguards ---
nltk_packages = ['punkt', 'stopwords']
for pkg in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg=='punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

ps = PorterStemmer()

# --- 1. Paths for Streamlit Cloud ---
BASE_DIR = os.path.dirname(__file__) # repo root folder
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
CSV_PATH = os.path.join(BASE_DIR, "spam.csv")

# --- 2. Preprocessing function ---
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = [i for i in text if i.isalnum()] # remove special chars
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text] # stemming
    return " ".join(text)

# --- 3. Load model & vectorizer ---
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

try:
    tfidf = pickle.load(open(VECT_PATH, 'rb'))
except Exception as e:
    st.warning(f"Vectorizer not loaded: {e}. Trying to fit from CSV...")
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, encoding='ISO-8859-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
        tfidf = pickle.load(open(VECT_PATH, 'rb')) if os.path.exists(VECT_PATH) else None
        if tfidf is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(max_features=3000)
            tfidf.fit(df['message'].apply(transform_text))
            with open(VECT_PATH, 'wb') as f:
                pickle.dump(tfidf, f)
            st.success("✅ Vectorizer fitted and saved locally!")
    else:
        st.error("❌ spam.csv not found in repo folder.")
        st.stop()

# --- 4. Streamlit UI ---
st.title("SMS/Email Spam Classifier")

input_sms = st.text_area("Enter your message:")

if st.button("Predict"):
    if not hasattr(tfidf, 'idf_'):
        st.error("Vectorizer not ready. Please ensure spam.csv exists and TF-IDF is fitted.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        st.header("🚨 SPAM" if result == 1 else "✅ NOT SPAM")

# --- 5. Debug info ---
if hasattr(tfidf, 'idf_'):
    st.sidebar.success("Model Status: Ready to go!")
else:
    st.sidebar.error("Model Status: Not Fitted")