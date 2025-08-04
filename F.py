
import pandas as pd
import re
from nltk.corpus import stopwords  # For stopwords removal
from nltk.stem.porter import PorterStemmer  # For stemming
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF Vectorization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier  # SGDClassifier for classification
from sklearn.metrics import accuracy_score


# In[2]:


# Load the datasets
train_df = pd.read_csv('dataset/train.csv')
fake_news_df = pd.read_csv('dataset/Fake.csv')
true_news_df = pd.read_csv('dataset/True.csv')


# In[3]:


# Label the datasets (0 for Real news, 1 for Fake news)
fake_news_df['label'] = 1  # Fake news is labeled as 1
true_news_df['label'] = 0  # Real news is labeled as 0


# In[4]:


# Combine the datasets into a single dataframe
news_df = pd.concat([true_news_df, fake_news_df], ignore_index=True)


# In[5]:


# Check for missing values
print(news_df.isna().sum())


# In[6]:


# Fill missing values with empty string
news_df = news_df.fillna(' ')


# In[7]:


# Combine the 'author' and 'title' columns into one 'content' column
# If 'author' column does not exist, use only 'title' or handle it accordingly
if 'author' in news_df.columns:
    news_df['content'] = news_df['author'] + " " + news_df['title']
else:
    news_df['content'] = news_df['title']


# In[8]:


# Initialize PorterStemmer for text preprocessing
ps = PorterStemmer()


# In[9]:


def stemming(content):
    # Remove non-alphabetic characters and convert to lowercase
    content = re.sub('[^a-zA-Z]', ' ', content).lower()
    
    # Tokenize and remove stopwords, then apply stemming
    content = [ps.stem(word) for word in content.split() if word not in stopwords.words('english')]
    
    # Join the processed words back into a string
    return ' '.join(content)


# In[10]:


# Apply stemming to the 'content' column
news_df['content'] = news_df['content'].apply(stemming)


# In[11]:


# Prepare feature and label arrays
X = news_df['content'].values  # Features (text content)
y = news_df['label'].values    # Labels (0 for real, 1 for fake)


# In[12]:


# Convert text data to numerical features using TF-IDF
vector = TfidfVectorizer()
X = vector.fit_transform(X)


# In[13]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


# In[14]:


# Check the shape of the training and testing sets
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# In[15]:


# Train the model using SGDClassifier
model = SGDClassifier(random_state=1)
model.fit(X_train, y_train)


# In[16]:


# Predicting on the training data
train_y_pred = model.predict(X_train)
print("Train accuracy:", accuracy_score(train_y_pred, y_train))


# In[17]:


# Predicting on the test data
test_y_pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(test_y_pred, y_test))


# In[18]:


# Prediction system for a sample from the test set
input_data = X_test[20]
prediction = model.predict(input_data)
if prediction[0] == 1:
    print('Fake news')
else:
    print('Real news')


# In[19]:


# Print content of the news for the prediction
print("Content of the news for the prediction:", news_df['content'].iloc[20])

