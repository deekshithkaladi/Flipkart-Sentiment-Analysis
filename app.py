import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved model
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("Flipkart Review Sentiment Analyzer")

review = st.text_area("Enter your review here")

if st.button("Predict Sentiment"):
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.success("Positive Review ")
    else:
        st.error("Negative Review ")
