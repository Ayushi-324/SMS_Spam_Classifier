# SMS Spam Classifier

An NLP-based machine learning project that classifies SMS messages as spam or not spam.

I built this project to understand the workflow deeply rather than just reproduce the output. The process included data cleaning, text preprocessing, feature extraction with TF-IDF, model training, model comparison, and turning the final model into a working Streamlit app.

## What this project does

- preprocesses raw SMS text
- transforms text into numerical features using TF-IDF
- predicts whether a message is spam or not spam
- provides a simple interactive web interface with Streamlit

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit

## Project Files

- `app.py` — Streamlit application
- `model.pkl` — trained machine learning model
- `vectorizer.pkl` — fitted TF-IDF vectorizer
- `spam.csv` — dataset used for training
- `SMS Spam Classifier.ipynb` — notebook used for experimentation, analysis, and training

## What I learned

This project strengthened my understanding of:

- text preprocessing for NLP tasks
- TF-IDF vectorization
- training and evaluating multiple classification models
- saving and loading trained ML components with pickle
- debugging environment, path, and deployment-related issues
- converting a notebook-based workflow into a usable web app

One of the biggest takeaways from this project was learning that building is not just about writing code — it is also about debugging environments, handling file structure properly, and knowing when to rebuild cleanly instead of forcing a broken setup.

## Run Locally

```bash
streamlit run app.py