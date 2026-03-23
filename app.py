import streamlit as st
import pickle
import os
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------
# 1️⃣ Safe NLTK download
# ---------------------
nltk_packages = ["punkt", "stopwords"]
for pack in nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{pack}" if pack=="punkt" else f"corpora/{pack}")
    except LookupError:
        nltk.download(pack)

ps = PorterStemmer()

# ---------------------
# 2️⃣ Text preprocessing
# ---------------------
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    # remove non-alphanumeric
    text = [i for i in text if i.isalnum()]
    # remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    # stemming
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# ---------------------
# 3️⃣ Load TF-IDF and Model
# ---------------------
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
CSV_PATH = "spam.csv"

# Load TF-IDF vectorizer safely
tfidf = None
if os.path.exists(VECTORIZER_PATH):
    try:
        with open(VECTORIZER_PATH, "rb") as f:
            tfidf = pickle.load(f)
    except Exception:
        tfidf = None

# Fallback: fit locally if pickle fails
if tfidf is None:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")
        df = df[['v1','v2']]
        tfidf = TfidfVectorizer(max_features=3000)
        tfidf.fit(df['v2'].apply(transform_text))
        # save pickle for next time
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(tfidf, f)
        st.info("✅ Vectorizer fitted locally and saved!")
    else:
        st.error(f"❌ {CSV_PATH} not found in repo. Upload it to GitHub.")
        st.stop()

# Load model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    st.error(f"❌ {MODEL_PATH} not found in repo. Upload it to GitHub.")
    st.stop()

# ---------------------
# 4️⃣ Streamlit UI
# ---------------------
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the Message:")

if st.button("Predict"):
    if tfidf is None or not hasattr(tfidf, "idf_"):
        st.error("⚠️ Vectorizer not ready. Ensure spam.csv exists and TF-IDF is fitted.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.header("🚨 SPAM")
        else:
            st.header("✅ NOT SPAM")

# ---------------------
# 5️⃣ Sidebar status
# ---------------------
if hasattr(tfidf, 'idf_'):
    st.sidebar.success("Model Status: Ready ✅")
else:
    st.sidebar.error("Model Status: Not Fitted ❌")
