import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Title
st.title("Sentiment Analysis with Customer Reviews (Improved Accuracy)")

# Input
user_input = st.text_area("Enter a customer review:")

# Analysis
if st.button("Analyze"):
    if user_input:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(user_input)
        compound_score = scores['compound']

        # Classify sentiment
        if compound_score >= 0.05:
            sentiment = "Positive"
        elif compound_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Output
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Compound Score:** {compound_score:.3f}")
        st.write(f"**Breakdown:** {scores}")

        # Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(user_input)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
