# Fake News Detection System

---

## Table of Contents
- [Project Overview](#project-overview)
- [Why Fake News Detection?](#why-fake-news-detection)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Features](#features)
- [How It Works](#how-it-works)
- [Usage Examples](#usage-examples)
- [Updating the Dataset & Model](#updating-the-dataset--model)
- [Future Improvements](#future-improvements)
- [Credits](#credits)
- [Running the Project](#running-the-project)

---

## Project Overview

This project is a **machine learning-based Fake News Detection system** that automatically classifies news articles as **Real** or **Fake**. It uses Natural Language Processing (NLP) techniques to preprocess news text and a trained classifier to detect misinformation.

The system can also fetch live news articles using public APIs, classify them, and present results in a user-friendly web application built with Streamlit.

---

## Why Fake News Detection?

- The rapid spread of misinformation impacts society, politics, health, and economies worldwide.
- Millions of news articles are published every day — manual verification is impossible.
- Automated detection systems help flag suspicious news quickly and efficiently.

---

## Technologies Used

| Technology             | Purpose                                             |
|-----------------------|-----------------------------------------------------|
| Python                | Programming language                                |
| Pandas                | Data loading and manipulation                       |
| NLTK                  | Text preprocessing (stopwords removal, stemming)   |
| scikit-learn          | Model training, evaluation, and prediction          |
| TfidfVectorizer       | Converting text into numerical features             |
| SGDClassifier         | Fast, scalable linear classification model          |
| Requests              | Fetch live news data from APIs                        |
| Joblib                | Save/load trained models and vectorizers             |
| Streamlit             | Web app framework for interactive UI                 |

---

## Project Workflow

1. **Data Collection:**  
   Collect labeled real and fake news datasets from public sources (e.g., Kaggle).

2. **Data Preprocessing:**  
   - Remove missing values.  
   - Clean text: lowercasing, removing punctuation, numbers.  
   - Remove stopwords and apply stemming.

3. **Feature Extraction:**  
   Transform processed text into TF-IDF vectors representing word importance.

4. **Model Training:**  
   Train an `SGDClassifier` on the labeled data and evaluate accuracy.

5. **Model Persistence:**  
   Save the trained model and TF-IDF vectorizer for future predictions.

6. **Live News Fetching & Classification:**  
   Use the NewsAPI to fetch latest news headlines, preprocess, vectorize, and classify.

7. **Interactive Web App:**  
   Allow users to paste news articles, get real-time classification, and see related news.

---

## Features

- **Accurate Fake News Detection:** Achieves ~94% test accuracy.  
- **Text Preprocessing Pipeline:** Handles noise, stopwords, and word stemming.  
- **Live News Classification:** Classify real-time headlines via public API.  
- **Interactive Streamlit App:** User-friendly interface for checking news authenticity.  
- **Model & Vectorizer Persistence:** Efficient re-use of trained artifacts.  
- **Expandable Dataset:** Easily update with new labeled news for better accuracy.

---

## How It Works

### Text Preprocessing

- Cleans the text by removing non-alphabetic characters and converting to lowercase.  
- Removes common stopwords (e.g., "the", "is", "at") that add little meaning.  
- Applies stemming to reduce words to their root form (e.g., “running” → “run”).

### Feature Extraction

- Converts text into TF-IDF vectors: words weighted by how important they are in the dataset.

### Model Training

- Uses `SGDClassifier` (linear model optimized with stochastic gradient descent) to classify news as fake or real.

### Prediction

- Incoming news text is preprocessed and transformed using the saved vectorizer.  
- The trained model predicts the label: 0 = Real News, 1 = Fake News.  
- Prediction results are displayed to the user.

---

## Usage Examples

### Train the Model

```bash
python train_model.py
