import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(page_title="IMDb Sentiment Analyzer", page_icon="🎬", layout="centered")

st.title("🎬 IMDb Movie Review Sentiment Analyzer")
st.write("Type a movie review below to see if the sentiment is **Positive** or **Negative**.")

# 1. Load the exported Model and Tokenizer (cached for performance)
@st.cache_resource
def load_assets():
    with open('tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    loaded_model = load_model('sentiment_lstm_model.h5')
    return loaded_tokenizer, loaded_model

try:
    tokenizer, model = load_assets()
except Exception as e:
    st.error("Error loading assets. Ensure 'tokenizer.pickle' and 'sentiment_lstm_model.h5' are in this directory.")
    st.stop()

# 2. Text preprocessing function (matching your notebook cleaning steps)
TAG_RE = re.compile(r'<[^>]+>')
def clean_text(text):
    sentence = TAG_RE.sub('', text) # Remove HTML tags
    sentence = re.sub(r'[^a-zA-Z]', ' ', sentence) # Remove punctuation & numbers
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence) # Single character removal
    sentence = re.sub(r'\s+', ' ', sentence) # Remove multiple spaces
    return sentence

# 3. User Interface
user_review = st.text_area("Your Review:", placeholder="Type something like: 'This movie was absolutely brilliant! loved the acting...'")

if st.button("Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Preprocess the raw input string
        cleaned_input = clean_text(user_review)
        
        # Tokenize text according to your exact extraction logic
        instance = tokenizer.texts_to_sequences([cleaned_input])
        
        flat_list = []
        for sublist in instance:
            for item in sublist:
                flat_list.append(item)
        flat_list = [flat_list]
        
        # Pad sequence to match training structure (maxlen = 100)
        maxlen = 100
        instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)
        
        # Run inference prediction
        prediction = model.predict(instance)[0][0]
        
        st.write("---")
        # Display outcome based on standard 0.5 threshold boundary
        if prediction >= 0.5:
            st.success(f"**Positive Sentiment** 🟢 (Confidence Score: {prediction*100:.2f}%)")
        else:
            st.error(f"**Negative Sentiment** 🔴 (Confidence Score: {(1 - prediction)*100:.2f}%)")