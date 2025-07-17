# app.py

import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Load the Trained Model ---
# This function loads our final, best-performing model.
# The @st.cache_resource decorator is excellent practice: it ensures the model
# is only loaded once when the app starts, not on every user interaction.
@st.cache_resource
def load_model():
    # --- MODIFIED ---
    # We now load the final model produced by our advanced training script.
    try:
        model = joblib.load('sentiment_model_final.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please run the training script first.")
        return None

model = load_model()

# --- Streamlit App Interface ---
st.title("Advanced Sentiment Analysis App")
st.write("This app uses a fine-tuned model to predict the sentiment of customer reviews.")

# Only show the main part of the app if the model was loaded successfully
if model:
    user_input = st.text_area("Enter a customer review to analyze:")

    if st.button("Analyze Sentiment"):
        if user_input:
            # 1. Use our trained pipeline to make a prediction.
            # The pipeline automatically handles lemmatization and vectorization.
            prediction = model.predict([user_input])
            sentiment = prediction[0]

            # --- MODIFIED: Handle confidence score carefully ---
            # The best model (LinearSVC) doesn't have predict_proba. We check for it.
            if hasattr(model, "predict_proba"):
                # This block runs if the model is LogisticRegression or MultinomialNB
                probability = model.predict_proba([user_input])
                confidence = probability.max()
                st.write(f"**Predicted Sentiment:** {sentiment.capitalize()}")
                st.success(f"**Confidence:** {confidence:.2f}") # Use st.success for good visibility
            else:
                # This block runs if the model is LinearSVC
                st.write(f"**Predicted Sentiment:** {sentiment.capitalize()}")
                st.info("Confidence scores are not available for the 'LinearSVC' model type.")

            # 2. Display Word Cloud (no changes needed here)
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
            st.warning("Please enter a review to analyze.")
else:
    st.warning("Model is not loaded. Cannot proceed.")