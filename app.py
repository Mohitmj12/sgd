import streamlit as st
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer


model = joblib.load('model/news_classification_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')


st.title('News Checker: Real or Fake?')


news_text = st.text_area('Paste your news article here:', '')

def fetch_news_by_topic(query, api_url="https://newsapi.org/v2/everything"):
    api_key = 'ea09ff9b3308472cb3eb810de2f429f8'  
    response = requests.get(f"{api_url}?q={query}&apiKey={api_key}")
    if response.status_code == 200:
        news_data = response.json()
        return news_data.get('articles', [])
    else:
        print("Failed to fetch news")
        return []


if st.button('Check News'):
    if news_text:
        
        news_vector = vectorizer.transform([news_text])

        
        prediction = model.predict(news_vector)

     
        if prediction == 0:  
            st.write('✅ The news is **REAL**')
        else:
            st.write('❌ The news is **FAKE**')
            
         
            st.write("Here are some other news articles related to the topic of your headline:")

           
            recommended_news = fetch_news_by_topic(news_text)
            
            if recommended_news:
                for article in recommended_news[:3]:  
                    st.write(f"**{article['title']}**")
                    st.write(f"Source: {article['source']['name']}")
                    st.write(f"Link: [Read More]({article['url']})")
                    st.write("\n")
            else:
                st.write("Sorry, we couldn't find related news.")
            
    else:
        st.write("Please enter a news article.")
