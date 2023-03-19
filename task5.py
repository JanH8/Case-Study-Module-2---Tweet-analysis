'''
# Task 5
use the trained Model to let the user verify his tweets
'''

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
from nltk.stem import WordNetLemmatizer

st.title('Task 5')


test = pd.read_csv('Data/test.csv')
test = test.drop('id', axis=1)


st.write('Let us have a look on the test data: ')
st.write(test)

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


def cleaning_tweets(tweet: str):
    tweet = tweet.replace('@user', '')
    tweet = ' '.join(tweet.split())
    tweet = tweet.lower()
    # searches for everything that is not a character between a and z, and not a whitespace
    tweet = re.sub(r'[^a-z\s]', '', tweet)

    splitted_tweet = tweet.split()
    result = []
    for x in splitted_tweet:
        if x not in stop:
            result.append(lemmatizer.lemmatize(x))  # add the lemmatized word

    return ' '.join(result)


test['tweet_cleaned'] = test['tweet'].apply(cleaning_tweets)

st.write('Now the tweets are cleaned:')
st.write(test)

clf = joblib.load('model.pkl')

st.write('now predict the values for the test values:')
test['prediction'] = clf.predict(test['tweet_cleaned'])
st.write(test)

st.header('Try your own tweets!')
st.text_input('test your own tweet:', key='own_tweet')
st.write('cleaned input: '+cleaning_tweets(st.session_state.own_tweet))
if st.session_state.own_tweet:
    result = clf.predict([cleaning_tweets(st.session_state.own_tweet)])
    if result == [1]:
        st.write('your tweet seems to contain hate speech!')
    else:
        st.write('your tweet seems to contain no hate speech!')
