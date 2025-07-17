import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
st.title("Sentiment Analysis with Customer Reviews")
user_input = st.text_area("Enter a customer review:")
if st.button("Analyze"):
    if user_input:
        blob = TextBlob(user_input)
        sentiment_score = blob.sentiment.polarity

        if sentiment_score > 0:
            sentiment = "Positive"
        elif sentiment_score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Polarity Score: {sentiment_score}")

        # Generate and display a word cloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(user_input)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)