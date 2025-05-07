import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Import dataset from Kaggle
data = pd.read_csv("fake_or_real_news.csv")

# Feature & Values
X = np.array(data["title"])
Y = np.array(data["label"])

# Initialize, fit and transform
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split and train
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train,Y_train)

# User Interface 
st.title("Fake News Detection System")
def fakenewsdetection():
    input = st.text_area("Enter a News Headline: ")
    if len(input) < 1:
        st.write("  ")
    else:
        sample = input
        data = vectorizer.transform([sample]).toarray()
        out = model.predict(data)[0]
        st.title(out)
fakenewsdetection()