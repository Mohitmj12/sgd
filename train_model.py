# train_model.py
import os
import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import nltk

# Download NLTK data
nltk.download('stopwords')

# Load datasets
fake_news_df = pd.read_csv('dataset/Fake.csv')
true_news_df = pd.read_csv('dataset/True.csv')

# Add labels
fake_news_df['label'] = 1
true_news_df['label'] = 0

# Merge
news_df = pd.concat([true_news_df, fake_news_df], ignore_index=True)
news_df = news_df.fillna(' ')

# Combine author and title
if 'author' in news_df.columns:
    news_df['content'] = news_df['author'] + ' ' + news_df['title']
else:
    news_df['content'] = news_df['title']

# Initialize tools
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocessing
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    return ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])

news_df['content'] = news_df['content'].apply(preprocess)

# Features and labels
X = news_df['content'].values
y = news_df['label'].values

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Train model
model = SGDClassifier(random_state=1)
model.fit(X_train, y_train)

# Accuracy
print(f"Train Accuracy: {accuracy_score(model.predict(X_train), y_train) * 100:.2f}%")
print(f"Test Accuracy: {accuracy_score(model.predict(X_test), y_test) * 100:.2f}%")

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/news_classification_model.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
print("✅ Model and vectorizer saved.")
