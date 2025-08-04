# streamlit_app.py
import streamlit as st
import joblib
import requests

# Load model and vectorizer
try:
    model = joblib.load('model/news_classification_model.pkl')
    vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
except:
    st.error("❌ Model files not found. Please run train_model.py first.")
    st.stop()

st.title('📰 News Checker: Real or Fake?')

# Input box
news_text = st.text_area('Paste your news headline or short article:', '')

# Fetch related news
def fetch_related_news(query):
    api_key = 'ea09ff9b3308472cb3eb810de2f429f8'  # Replace with your own key
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url)
    return response.json().get('articles', []) if response.status_code == 200 else []

# On click
if st.button('Check News'):
    if news_text.strip():
        transformed = vectorizer.transform([news_text])
        prediction = model.predict(transformed)

        if prediction[0] == 0:
            st.success("✅ The news is **REAL**.")
        else:
            st.error("❌ The news is **FAKE**.")

            st.markdown("---")
            st.markdown("#### 🔎 Related News Articles")
            related = fetch_related_news(news_text)
            if related:
                for article in related[:3]:
                    st.write(f"**{article['title']}**")
                    st.write(f"🗞️ Source: {article['source']['name']}")
                    st.write(f"[Read More]({article['url']})")
            else:
                st.info("No related articles found.")
    else:
        st.warning("Please enter news text to analyze.")
