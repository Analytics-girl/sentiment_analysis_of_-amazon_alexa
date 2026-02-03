# 1. Save correct files in Colab (pick best model):
pickle.dump(model_xgb, open('model_xgb.pkl', 'wb'))  # Rename to match app
pickle.dump(cv, open('vectorizer.pkl', 'wb'))
files.download('model_xgb.pkl')  # etc.

# 2. Add preprocessing function to Streamlit BEFORE vectorizer:
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in STOPWORDS]
    return ' '.join(text)

# In predict button:
if st.button("Predict"):
    processed_text = preprocess(user_input)
    text_vector = vectorizer.transform([processed_text])  # Use PROCESSED text
    # ... rest same
