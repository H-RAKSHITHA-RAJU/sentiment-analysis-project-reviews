# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib # Used to save our model

# 1. Load Data
df = pd.read_csv('IMDB_Dataset.csv')

# Drop any rows with missing values
df.dropna(inplace=True)

# Define features (X) and target (y)
X = df['review']
y = df['sentiment']

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Create a model pipeline
# A pipeline makes it easy to chain steps together:
# a. TfidfVectorizer: Converts text into numerical features.
# b. MultinomialNB: A Naive Bayes classifier suitable for text.
model = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# 4. Train the model
model.fit(X_train, y_train)

# 5. Evaluate the model (optional but good practice)
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 6. Save the trained model to a file
joblib.dump(model, 'sentiment_model.pkl')

print("Model trained and saved as sentiment_model.pkl")