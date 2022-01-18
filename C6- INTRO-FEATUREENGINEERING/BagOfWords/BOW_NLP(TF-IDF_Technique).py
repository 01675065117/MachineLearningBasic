# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 11:20:34 2021

@author: Admin
"""

# Import LIBS
import numpy as np
import pandas as pd

import re
import nltk
nltk.download('stopwords')
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer #TF function
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
#-------------------------------------------------------

def get_data():
    data_imdb = pd.read_csv("media\\sentiment_labelled_sentences\\imdb_labelled.txt", delimiter='\t', header = None)
    data_imdb.columns = ["Review_text", "Review class"]
    
    data_amazon = pd.read_csv("media\\sentiment_labelled_sentences\\amazon_cells_labelled.txt", delimiter='\t', header = None)
    data_amazon.columns = ["Review_text", "Review class"]
    
    data_yelp = pd.read_csv("media\\sentiment_labelled_sentences\\yelp_labelled.txt", delimiter='\t', header = None)
    data_yelp.columns = ["Review_text", "Review class"]
    
    data = pd.concat([data_imdb,data_amazon,data_yelp])
    
    return data




def clean_text(df):
    all_reviews = list()
    lines = df["Review_text"].values.tolist()
    for text in lines:
        text = text.lower()
        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = pattern.sub('', text)
        text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text) #Remove special characters
        tokens = word_tokenize(text) #Word_tokenize is a function in Python that splits a given sentence into words using the NLTK library.
        table = str.maketrans('','', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        PS = PorterStemmer() #Check the words and make it original || for example "cats" -> "cat", "troubled" -> "troubl"
        words = [PS.stem(w) for w in words if not w in stop_words]
        words = ' '.join(words)
        all_reviews.append(words)
    return all_reviews

def split_data(all_reviews,Data):
    TV =TfidfVectorizer(min_df=3)#Fix CV -> TF
    x = TV.fit_transform(all_reviews).toarray()#fit các vector với all_reviews
    y = Data[["Review class"]].values
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    return X_train, X_test, Y_train, Y_test


def decision_tree (X_train, X_test, Y_train, Y_test):
    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(Y_test, y_pred))
    print(f1_score(Y_test, y_pred))
    print(precision_score(Y_test, y_pred))

def naive_bayes (X_train, X_test, Y_train, Y_test):
    model = GaussianNB()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(Y_test, y_pred))
    print(f1_score(Y_test, y_pred))
    print(precision_score(Y_test, y_pred))

def main():
    Data = get_data()
    all_reviews = clean_text(Data)
    X_train, X_test, Y_train, Y_test = split_data(all_reviews,Data)
    print('decision_tree')
    decision_tree (X_train, X_test, Y_train, Y_test)
    print('-------------------------------------------------')
    print('naive bayes')
    naive_bayes (X_train, X_test, Y_train, Y_test)

main()






















