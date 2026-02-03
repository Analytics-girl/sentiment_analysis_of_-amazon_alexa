import streamlit as st
import pickle

# Load model & vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("Sentiment Analysis App")

user_input = st.text_area("Enter text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)

        if prediction[0] == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")
