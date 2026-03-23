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
from sklearn.exceptions import NotFittedError

# -----------------------------
# 1. NLTK safeguards for Streamlit
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
# 2. Text preprocessing function
# -----------------------------
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    
    # Remove non-alphanumeric tokens
    tokens = [t for t in tokens if t.isalnum()]
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    
    # Stemming
    tokens = [ps.stem(t) for t in tokens]
    
    return " ".join(tokens)

# -----------------------------
# 3. Load model and vectorizer
# -----------------------------
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")

try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading vectorizer.pkl: {e}")

# Ensure TF-IDF is fitted
if 'tfidf' in locals():
    try:
        if not hasattr(tfidf, 'idf_'):
            st.warning("Vectorizer not fitted. Fitting locally using spam.csv...")
            if os.path.exists('spam.csv'):
                df = pd.read_csv('spam.csv', encoding='ISO-8859-1')[['v1','v2']]
                tfidf = TfidfVectorizer(max_features=3000)
                tfidf.fit(df['v2'].apply(transform_text))
                with open('vectorizer.pkl', 'wb') as f:
                    pickle.dump(tfidf, f)
                st.success("✅ TF-IDF fitted and saved locally")
            else:
                st.error("❌ spam.csv not found. Cannot fit vectorizer.")
    except Exception as e:
        st.error(f"TF-IDF check error: {e}")

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.title("📩 SMS Spam Classifier")

input_sms = st.text_area("Enter your message here:")

if st.button("Predict"):
    if not hasattr(tfidf, 'idf_'):
        st.error("Vectorizer is not ready. Please ensure spam.csv exists and TF-IDF is fitted.")
    else:
        try:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            
            if result == 1:
                st.header("🚨 SPAM")
            else:
                st.header("✅ NOT SPAM")
        except NotFittedError:
            st.error("Vectorizer is still not fitted. Check spam.csv and rebuild vectorizer.")
        except Exception as e:
            st.error(f"Unexpected error during prediction: {e}")

# -----------------------------
# 5. Sidebar: Model status
# -----------------------------
if 'tfidf' in locals() and hasattr(tfidf, 'idf_'):
    st.sidebar.success("Model Status: Ready ✅")
else:
    st.sidebar.error("Model Status: Not fitted ❌")

