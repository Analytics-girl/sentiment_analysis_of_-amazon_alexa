import streamlit as st
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------
# Load models (cached for speed)
# -------------------------------
@st.cache_resource
def load_models():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, vectorizer, scaler

# -------------------------------
# Download NLTK stopwords
# -------------------------------
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

# -------------------------------
# Load models
# -------------------------------
model, vectorizer, scaler = load_models()

# -------------------------------
# Preprocessing function
# -------------------------------
stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def preprocess(text):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    # Stemming and removing stopwords
    text = [stemmer.stem(word) for word in text if word not in STOPWORDS]
    return ' '.join(text)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🗣️ Amazon Alexa Sentiment Analysis")
st.caption("Powered by XGBoost - 94%+ accuracy")

user_input = st.text_area(
    "Enter review text", 
    height=150, 
    placeholder="e.g., 'I love this device!'"
)

if st.button("🔮 Predict Sentiment"):
    if user_input.strip():
        processed = preprocess(user_input)

        # Transform text to feature vector
        vector = vectorizer.transform([processed])

        # Convert sparse matrix to dense before scaling
        vector_dense = vector.toarray()
        vector_scaled = scaler.transform(vector_dense)

        # Prediction
        pred = model.predict(vector_scaled)[0]
        prob = model.predict_proba(vector_scaled)[0]

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            if pred == 1:
                st.success("✅ **Positive** 😊")
            else:
                st.error("❌ **Negative** 😞")
        with col2:
            st.info(f"**Confidence**: {max(prob):.1%}")

        st.write(f"**Processed text**: `{processed}`")
    else:
        st.warning("⚠️ Please enter some text")
