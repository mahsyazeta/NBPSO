import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# Initialize Sastrawi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load stopwords
list_stopwords = stopwords.words('indonesian')
list_stopwords.extend([
    "yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang', 
    'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
    'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt', '&amp', 'yah'
])
txt_stopword = pd.read_csv("stopword.txt", names=["stopwords"], header=None)
list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
list_stopwords = set(list_stopwords)

def preprocess_data(df):
    # Drop rows with missing values
    df = df.dropna(subset=['Text Tweet', 'Sentiment'])
    
    # Cleansing
    df['Text Tweet'] = df['Text Tweet'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    
    # Case folding
    df['Text Tweet'] = df['Text Tweet'].str.lower()
    
    # Formalisasi dan stemming
    df['Text Tweet'] = df['Text Tweet'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    
    # Remove stopwords
    df['Text Tweet'] = df['Text Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in list_stopwords]))
    
    # Tokenizing (CountVectorizer will handle tokenizing in the classification phase)
    
    return df
