import requests
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = joblib.load('model/news_classification_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Function to fetch news from an API
def fetch_news(api_url="https://newsapi.org/v2/top-headlines?country=us"):
    # Hardcode the API key directly in the code
    api_key = 'ea09ff9b3308472cb3eb810de2f429f8'  # Replace with your actual key
    response = requests.get(f"{api_url}&apiKey={api_key}")
    if response.status_code == 200:
        news_data = response.json()
        return news_data.get('articles', [])
    else:
        print("Failed to fetch news")
        return []

# Function to preprocess the news title
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    # Add more preprocessing steps as needed (e.g., remove punctuation, numbers)
    return text

# Function to classify news using the trained model
def classify_news(news_articles):
    for article in news_articles:
        title = article.get('title', 'No Title Provided')  # Ensure title exists
        preprocessed_title = preprocess_text(title)

        # Convert the preprocessed title into a vector
        X_test_title = vectorizer.transform([preprocessed_title])

        # Make prediction using the trained model
        prediction = model.predict(X_test_title)
        prediction_result = "Real News" if prediction[0] == 0 else "Fake News"  # Adjust labels as needed

        # Print the result
        print(f"Title: {title}")
        print(f"Prediction: {prediction_result}\n")

# Main function to fetch and classify live news
def main():
    news_articles = fetch_news()  # Fetch live news
    if news_articles:
        classify_news(news_articles)  # Classify the fetched news

if __name__ == "__main__":
    main()
