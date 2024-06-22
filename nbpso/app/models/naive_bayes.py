import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def classify(df, model_type, params=None):
    df['label_sentimen'] = df['Sentiment'].apply(lambda x: 1 if x == "positive" else 0)
    
    vectorizer = TfidfVectorizer(smooth_idf=True)
    X = vectorizer.fit_transform(df['Text Tweet'])
    y = df['label_sentimen']

    if model_type == 'multinomial':
        model = MultinomialNB()
    elif model_type == 'bernoulli':
        model = BernoulliNB()
    elif model_type == 'gaussian':
        model = GaussianNB()
    else:
        raise ValueError("Invalid model type. Choose 'multinomial', 'bernoulli', or 'gaussian'.")

    if params:
        selected_indices = params.get('selected_indices')
        if selected_indices:
            X = X[:, selected_indices]

    kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    scores = []
    confusion_matrices = []
    metrics_list = []

    # Inisialisasi DataFrame results sebelum loop validasi silang dimulai
    results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        X_train, X_test = X[train_idx], X[val_idx]
        y_train, y_test = y[train_idx], y[val_idx]

        model.fit(X_train.toarray(), y_train)
        y_pred = model.predict(X_test.toarray())

        scores.append(accuracy_score(y_test, y_pred))
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

        # Tambahkan hasil prediksi ke dalam list results
        for i, pred in enumerate(y_pred):
            results.append({'Fold': fold, 'Predicted Sentiment': pred})

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        metrics_list.append({'accuracy': accuracy, 'precision': precision, 'recall': recall})

    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in metrics_list]) * 100,
        'precision': np.mean([m['precision'] for m in metrics_list]) * 100,
        'recall': np.mean([m['recall'] for m in metrics_list]) * 100
    }

    avg_conf_matrix = np.mean(confusion_matrices, axis=0)
    results_df = pd.DataFrame(results)
    sentiment_distribution = df['Sentiment'].value_counts(normalize=True) * 100
    plt.figure(figsize=(6, 6))
    sentiment_distribution.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Sentiment Distribution')
    plt.ylabel('')
    sentiment_dist_path = 'static/sentiment_distribution.png'
    plt.savefig(sentiment_dist_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Predicted Sentiment', data=results_df)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    sentiment_dist_path = os.path.join('static', 'sentiment_distribution.png')
    plt.savefig(sentiment_dist_path)
    plt.close()
    return results_df, avg_conf_matrix, avg_metrics, sentiment_dist_path

def predict_text(text, preprocessed_df):
    # Fit the vectorizer and model with preprocessed data
    vectorizer = TfidfVectorizer(smooth_idf=True)
    X = vectorizer.fit_transform(preprocessed_df['Text Tweet'])
    y = preprocessed_df['Sentiment'].apply(lambda x: 1 if x == "positive" else 0)
    
    model = MultinomialNB()  # Use the Multinomial Naive Bayes model
    model.fit(X, y)
    
    # Transform the input text using the fitted vectorizer
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    
    return "positive" if prediction[0] == 1 else "negative"
