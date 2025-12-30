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

#  page ui 
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")
st.title("📊 Twitter Sentiment Analysis with Live Prediction")
st.sidebar.header("Controls")

#file upload
if "df" not in st.session_state:
    data_train = pd.read_csv('train.csv', encoding='latin1')
    data_test = pd.read_csv('test.csv', encoding='latin1')
    st.session_state.df = pd.concat([data_train, data_test], ignore_index=True)

df = st.session_state.df

#dataset head
if st.sidebar.button("👀 Show Dataset Head", key="head_btn"):
    st.write("### Dataset Head")
    st.dataframe(df.head())

#dataset info
if st.sidebar.button("📊 Show Dataset Info", key="info_btn"):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.write("### Dataset Info")
    st.text(info_str)

# remove irrelevant column
cols_to_drop_manual = st.sidebar.multiselect(
    "Select columns to drop manually",
    df.columns
)

if st.sidebar.button("❌ Drop Selected Columns", key="drop_manual"):
    if cols_to_drop_manual:
        df = df.drop(columns=cols_to_drop_manual)
        st.session_state.df = df  # save changes
        st.success("Selected columns removed")
    else:
        st.warning("No columns selected to drop")

# check null value 
st.sidebar.write("Null Values")
if st.sidebar.button("Check Null Values"):
  df = st.session_state.df
  null_counts = df.isnull().sum()
  st.write("### Null Values per Column")
  st.dataframe(null_counts)

# remove row null value
if st.sidebar.button("Remove Null Rows"):
    df = st.session_state.df
    df = df.dropna()
    st.session_state.df = df
    st.success("Null rows removed")

# display bar chart
st.sidebar.write("Bar Chart of Selected Column")
bar_col = st.sidebar.selectbox("Select Column for Bar Chart", df.columns)

if st.sidebar.button("Show Bar Chart"):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(data=df, x=bar_col, ax=ax, palette='inferno')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# text cleaning method
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

# test cleaning 
st.sidebar.write("Text Cleaning")
text_col_clean = st.sidebar.selectbox("Select Text Column to clean", df.columns)

if st.sidebar.button("🧽 Clean Text", key="clean_text"):
    if text_col_clean:
            df["processed_text"] = df[text_col_clean].astype(str).apply(preprocess_text)
            st.session_state.df = df  # save changes
            st.success("Selected columns clean")
    else:
            st.warning("No columns selected to clean")

# model training - tf-idf, logistic and random
st.sidebar.write("Training")
if "processed_text" in df.columns:
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])
    
    if st.sidebar.button("🚀 Train Model"):
        df_cleaned = st.session_state.df
        X = df_cleaned["processed_text"]
        y = df_cleaned[target_col]

        # Encode target if text
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        if y.dtype == 'object':
            y = y.map(label_map)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # TF-IDF
        vectorizer = TfidfVectorizer()
        XV_train = vectorizer.fit_transform(X_train)
        XV_test = vectorizer.transform(X_test)

        # Model
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = RandomForestClassifier(random_state=0)

        model.fit(XV_train, y_train)

        # Store in session
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.XV_test = XV_test
        st.session_state.y_test = y_test

        st.success(f"{model_choice} trained successfully")

# model evaluation
if "model" in st.session_state:
    st.write("### Model Evaluation")
    model = st.session_state.model
    XV_test = st.session_state.XV_test
    y_test = st.session_state.y_test

    y_pred = model.predict(XV_test)
    st.write("Accuracy:", model.score(XV_test, y_test))
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(6,6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    st.pyplot(fig)

# real-time
st.subheader("🔴 Real-Time Prediction")
user_text = st.text_area("Enter text to predict sentiment")

if st.button("Predict Sentiment"):
    if "model" in st.session_state and "vectorizer" in st.session_state:
        # Preprocess user input
        processed_input = preprocess_text(user_text)
        vec = st.session_state.vectorizer.transform([processed_input])
        pred = st.session_state.model.predict(vec)[0]
        label_map_inv = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.success(f"Prediction: **{label_map_inv[pred]}**")
    else:
        st.warning("Please train a model first!")



