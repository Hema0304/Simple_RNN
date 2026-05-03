import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
reverse_wor_index = {values:key for key , values in word_index.items()}
model = load_model('simple_rnn_imdb.keras')

def decode_review(encoded_review):
    return ' '.join([reverse_wor_index.get(i - 3,'?') for i in encoded_review])


def preprocess_text(text):
    words=text.lower().split()
    encoded_review = [word_index.get(word,2) for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

##prediction function

def predict_sentiment(review):
    preprocess_input=preprocess_text(review)

    prediction = model.predict(preprocess_input) 
    score = float(prediction[0][0])
    sentiment = "Positive" if score > 0.5 else "Negative"
    
    return sentiment, score
#streamlit app
st.set_page_config(page_title="IMDB Sentiment Analysis")
st.title("IMDB Movie Review Sentiment Analysis")
st.write('Enter a movie review to classify it is positive or negative')
user_input = st.text_area('Movie Review')
 
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a movie review")
    else:
        sentiment, score = predict_sentiment(user_input)
        
        st.success(f"Sentiment: {sentiment}")
        st.write(f"Prediction score: {score:.4f}")
        
