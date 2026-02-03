import streamlit as st
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Download nltk data once
nltk.download('stopwords', quiet=True)

# Load your saved files (from Colab downloads)
with open("model.pkl", "rb") as f:    # <- Rename your app to load this
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing function (exactly like training)
stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in STOPWORDS]
    return ' '.join(text)

st.title("Sentiment Analysis")
user_input = st.text_area("Enter text")

if st.button("Predict"):
    if user_input.strip():
        processed = preprocess(user_input)
        vector = vectorizer.transform([processed])
        vector_scaled = scaler.transform(vector)  # <- Scale too!
        pred = model.predict(vector_scaled)[0]
        st.success("Positive 😊") if pred == 1 else st.error("Negative 😞")
