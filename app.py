import gradio as gr
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# 2. PREPROCESSING FUNCTION
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    
    y = [i for i in text if i.isalnum()]

    stop_words = set(stopwords.words('english'))
    y = [i for i in y if i not in stop_words and i not in string.punctuation]

    y = [ps.stem(i) for i in y]

    return " ".join(y)

# 3. LOAD MODELS
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# 4. THE PREDICT FUNCTION
def predict_spam(input_sms):
    if not input_sms.strip():
        return "Please enter a message."
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    return "🚨 SPAM" if result == 1 else "✅ NOT SPAM"

# 5. THE GRADIO INTERFACE 
demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(label="Enter Message"),
    outputs=gr.Label(label="Result"),
    title="SMS Spam Classifier"
)

if __name__ == "__main__":
    demo.launch()
