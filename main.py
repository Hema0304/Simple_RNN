import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_wor_index = {values:key for key , values in word_index.items()}
model = load_model('simple_rnn_imdb.keras')

def decode_review(encoded_review):
    return ' '.join([reverse_wor_index.get(i - 3,'?') for i in encoded_review])


def preprocess_text(text):
    words=text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

##prediction function

def predict_sentiment(review):
    preprocess_input=preprocess_text(review)

    prediction = model.predict(preprocess_input) 
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment , prediction[0][0]

#streamlit app
import streamlit as st 
st.title("IMDB Movie Review Sentiment Analysis")
st.write('Enter a movie review to classify it is positive or negative')
user_input = st.text_area('Movie Review')
 
if st.button("Classify"):
    if user_input.strip() == "":
        st.write("Please enter a movie review")
    else:
        preprocess_input = preprocess_text(user_input)
        prediction = model.predict(preprocess_input)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction score: {prediction[0][0]}")