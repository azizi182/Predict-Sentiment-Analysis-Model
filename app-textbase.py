import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
import string
import io

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# A 
# Load dataset
data_train = pd.read_csv('train.csv',encoding= 'latin1')
data_test = pd.read_csv('test.csv',encoding= 'latin1')

#print data
data_train.info()
data_test.info()

# combine all dataset, to make a rule of prediction. more dataset more accurate
df = pd.concat([data_train,data_test])
print()
df.info()

# B - preprocessing
# 1. handle null value
# to know is a dataset have null
df.isnull().sum()

# remove all null row
df = df.dropna()
print(df.isnull().sum())
print()
# print again dataset
df.info()

# 2. count sentiment
# to know a negative, positive and neutral sentiment

print("count of total sentiment")
print(df['sentiment'].value_counts())
print()

plt.figure(figsize=(8, 5))
plt.title('Bar Chart (Neutral vs Positive vs Negative)')
sns.countplot(data=df, x='sentiment', hue='sentiment', palette='inferno')
plt.show()

# 3. remove inrrelevant column
#df=df.drop(columns=['textID','Time of Tweet', 'Age of User', 'Country', 'Population -2020', 'Land Area (Km²)', 'Density (P/Km²)'])
df.head()
#df.info()

# 4. text cleaning
def clean_text(text):

    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove user mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags (keeping the text after #)
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_text(text):

    if not text:
        return ""

    # Clean text
    text = clean_text(text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into text
    processed_text = ' '.join(tokens)

    return processed_text

# NLTK downloads 
# Ensure text column is string
df['text'] = df['text'].astype(str)

# Apply preprocessing
df_cleaned = df.copy()
# make a new column for a result of text cleaning
df_cleaned['processed_text'] = df_cleaned['selected_text'].apply(preprocess_text)
df_cleaned


# C. Training
# 1. TF-IDF

X=df_cleaned['processed_text']
y= df_cleaned['sentiment']

# change sentiment to int
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
y = y.map(label_map)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# implement a TF-IDF
vectorization = TfidfVectorizer()
XV_train = vectorization.fit_transform(X_train)
XV_test = vectorization.transform(X_test)


# do a baseline- what the proportion of the most
# frequent sentiment in a dataset
score_baseline = df['sentiment'].value_counts(normalize=True).max()
score_baseline

# 2. Logistic
#training logistic
logisticmodel = LogisticRegression(max_iter=1000)
logisticmodel.fit(XV_train,y_train)

#testing
pred_logistic = logisticmodel.predict(XV_test)

#accurancy
accurancy = logisticmodel.score(XV_test,y_test)

#print
print(f"Accuracy: {accurancy:.2f}")
print(classification_report(y_test, pred_logistic))

#confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, pred_logistic);

# 3. Random Forest
#training
forestmodel = RandomForestClassifier(random_state=0)
forestmodel.fit(XV_train,y_train)

#testing
pred_forest = forestmodel.predict(XV_test)

#accurancy
accurancy = forestmodel.score(XV_test,y_test)

#print
print(f"Accuracy: {accurancy:.2f}")
print(classification_report(y_test, pred_forest))

#confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, pred_forest);


