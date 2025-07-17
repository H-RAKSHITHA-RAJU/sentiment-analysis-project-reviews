# app.py

import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Load the Trained Model ---
# We load the model once at the start of the app
# The @st.cache_resource decorator ensures this function is only run once
@st.cache_resource
def load_model():
    model = joblib.load('sentiment_model.pkl')
    return model

model = load_model()

# --- Streamlit App Interface ---
st.title("Custom Sentiment Analysis App")
st.write("This app uses a model trained on a specific customer review dataset.")

user_input = st.text_area("Enter a customer review to analyze:")

if st.button("Analyze Sentiment"):
    if user_input:
        # 1. Use our trained model to make a prediction
        # The model expects a list or iterable of texts, so we pass [user_input]
        prediction = model.predict([user_input])
        probability = model.predict_proba([user_input])

        sentiment = prediction[0]
        confidence = probability.max()

        # Display the results
        st.write(f"**Predicted Sentiment:** {sentiment.capitalize()}")
        st.write(f"**Confidence:** {confidence:.2f}")

        # 2. Display Word Cloud (optional, but nice)
        st.write("---")
        st.write("### Word Cloud for the Review")
        try:
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(user_input)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        except ValueError:
            st.write("Could not generate a word cloud for this input.")

    else:
        st.write("Please enter a review to analyze.")