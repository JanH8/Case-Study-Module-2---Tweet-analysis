"""
# Task 5
use the trained Model to let the user verify his tweets
"""

import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
import joblib


train = pd.read_csv("Data/train.csv")
train = train.drop("id", axis=1)


st.write("Let's have a look on the training data:")
st.write(train)

stop = stopwords.words("english")


def cleaning_tweets(tweet: str):
    tweet = tweet.replace("#", "")
    tweet = tweet.replace("@user", "")
    tweet = " ".join(tweet.split())
    # searches for everything that is not a character between a and z, not a number and not a whitespace
    tweet = re.sub(r'[^A-Za-z0-9\s]', '', tweet)

    splitted_tweet = tweet.split()
    result = []
    for x in splitted_tweet:
        if x not in stop:
            result.append(x)

    return " ".join(result)


train["tweet"] = train["tweet"].apply(cleaning_tweets)

st.write("Now the tweets are cleaned:")
st.write(train)


# param stratify: keep the proportion of values in that set
x_train, x_test, y_train, y_test = train_test_split(
    train["tweet"], train["label"], test_size=0.3, stratify=train["label"])


porter = PorterStemmer()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=True,
                        preprocessor=None,
                        token_pattern=None)


param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__tokenizer': [tokenizer_porter],
               'clf__penalty': ['l1'],
               'clf__C': [10.0]},
              ]


clf = joblib.load("model.pkl")

st.write("now predict the values for the test values:")
predictions = clf.predict(x_test)
df1 = pd.DataFrame({"tweets": x_test, "labels": predictions})
st.write(df1)

st.text_input("test your own tweet:", key="own_tweet")
st.write("cleaned input: "+cleaning_tweets(st.session_state.own_tweet))
if st.session_state.own_tweet:
    result = clf.predict([cleaning_tweets(st.session_state.own_tweet)])
    if result == [1]:
        st.write("your tweet seems to contain hate speech!")
    else:
        st.write("your tweet seems to contain no hate speech!")
