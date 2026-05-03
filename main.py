import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# -------------------------------
# Load IMDB word index correctly
# -------------------------------
word_index = imdb.get_word_index()

# Shift indices (must match training)
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# -------------------------------
# Load trained model
# -------------------------------
model = load_model("simple_rnn_imdb.keras")

# -------------------------------
# Preprocess input text
# -------------------------------
def preprocess_text(text):
    words = text.lower().split()

    encoded = []
    for word in words:
        idx = word_index.get(word, 2)  # unknown → 2

        # Restrict vocab (VERY IMPORTANT)
        if idx >= 10000:
            idx = 2

        encoded.append(idx)

    padded = sequence.pad_sequences([encoded], maxlen=500)

    # Handle RNN input shape automatically
    if len(model.input_shape) == 3:
        padded = padded.reshape(1, 500, 1)

    return padded

# -------------------------------
# Predict sentiment
# -------------------------------
def predict_sentiment(review):
    processed = preprocess_text(review)

    prediction = model.predict(processed)

    score = float(prediction[0][0])
    sentiment = "Positive" if score > 0.5 else "Negative"

    return sentiment, score

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="IMDB Sentiment Analysis")
st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write("Enter a movie review to classify sentiment.")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a movie review")
    else:
        sentiment, score = predict_sentiment(user_input)

        st.success(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score:.4f}")
