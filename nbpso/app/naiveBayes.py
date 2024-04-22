import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score


def multinomial_naive_bayes(file_path):
    df = pd.read_csv(file_path)  # Assuming the data is in CSV format
    X = df['tweet_tokens_stemmed']
    y = df['Sentiment']

    tfidf_model = TfidfVectorizer(smooth_idf=False)
    X_tfidf = tfidf_model.fit_transform(X)

    mnb = MultinomialNB()
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(mnb, X_tfidf, y, cv=kfold)
    accuracy = np.mean(scores)
    return accuracy * 100

def bernoulli_naive_bayes(file_path):
    df = pd.read_csv(file_path)  # Assuming the data is in CSV format
    X = df['tweet_tokens_stemmed']
    y = df['Sentiment']

    tfidf_model = TfidfVectorizer(smooth_idf=False)
    X_tfidf = tfidf_model.fit_transform(X)

    bnb = BernoulliNB()
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(bnb, X_tfidf, y, cv=kfold)
    accuracy = np.mean(scores)
    return accuracy * 100

def gaussian_naive_bayes(file_path):
    df = pd.read_csv(file_path)  # Assuming the data is in CSV format
    X = df['tweet_tokens_stemmed']
    y = df['Sentiment']

    tfidf_model = TfidfVectorizer(smooth_idf=False)
    X_tfidf = tfidf_model.fit_transform(X).toarray()

    gnb = GaussianNB()
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(gnb, X_tfidf, y, cv=kfold)
    accuracy = np.mean(scores)
    return accuracy * 100
