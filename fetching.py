# live_classify.py
import requests
import joblib

# Load model and vectorizer
model = joblib.load('model/news_classification_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

def fetch_news(api_url="https://newsapi.org/v2/top-headlines?country=us"):
    api_key = 'ea09ff9b3308472cb3eb810de2f429f8'  # Replace with your API key
    response = requests.get(f"{api_url}&apiKey={api_key}")
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        print("Failed to fetch news.")
        return []

def preprocess(text):
    return text.lower()

def classify_news(articles):
    for article in articles:
        title = article.get('title', 'No Title')
        preprocessed = preprocess(title)
        vector = vectorizer.transform([preprocessed])
        pred = model.predict(vector)
        label = "Real News ✅" if pred[0] == 0 else "Fake News ❌"
        print(f"\nTitle: {title}\nPrediction: {label}\n")

def main():
    news = fetch_news()
    if news:
        classify_news(news)

if __name__ == "__main__":
    main()
