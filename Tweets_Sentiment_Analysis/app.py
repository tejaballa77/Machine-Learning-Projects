import streamlit as st
import pandas as pd
import pickle

# Load model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("✈️ Tweet Sentiment Prediction App")
st.write("Predict if a tweet is Positive, Negative, or Neutral.")

# Sidebar for input
tweet_input = st.text_area("Enter a Tweet:")

if st.button("Predict"):
    # Preprocess input: lowercase, remove punctuation (same as training)
    import re, string
    cleaned = tweet_input.lower()
    cleaned = re.sub(f"[{string.punctuation}]", "", cleaned)

    # Lemmatize
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    cleaned = " ".join([lemmatizer.lemmatize(w) for w in cleaned.split()])

    # Convert to vector (TF-IDF)
    import pickle
    with open("tfidf_vectorizer.pkl", "rb") as f:  # TF-IDF used during training
        vectorizer = pickle.load(f)
    vectorized_input = vectorizer.transform([cleaned])

    # Prediction
    prediction = model.predict(vectorized_input)
    st.subheader("Predicted Sentiment:")
    st.write(prediction[0])
