import re
import dill
import nltk
import faiss
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

# -------------------------------
# Load Datasets
# -------------------------------
anime_data = pd.read_csv('Data/anime-dataset-2023.csv')
movie_data = pd.read_csv('Data/TMDB_movie_dataset_v11.csv')

# Anime dataset
anime_data = anime_data.iloc[:, [2, 4, 5, 6, 23]]
anime_data = anime_data.rename(columns={
    'English name': 'title',
    'Score': 'rating',
    'Genres': 'genres',
    'Synopsis': 'description',
    'Image URL': 'img_url'
})

# Movie dataset
movie_data = movie_data.iloc[:, [1, 2, 19, 15, 17]]
movie_data = movie_data.rename(columns={
    'vote_average': 'rating',
    'overview': 'description',
    'poster_path': 'image_url'
})
movie_data['rating'] = np.round(movie_data['rating'], 2)

# Cleaning
anime_data['rating'] = pd.to_numeric(anime_data['rating'], errors='coerce')
anime_data = anime_data.drop_duplicates(subset=anime_data.columns.difference(['rating','description'])).reset_index(drop=True)
movie_data = movie_data.drop_duplicates(subset=movie_data.columns.difference(['rating'])).reset_index(drop=True)
anime_data.dropna(subset=['rating'], inplace=True)
movie_data.dropna(subset=movie_data.columns.difference(['rating']), inplace=True)
anime_data = anime_data[anime_data['title'] != 'UNKNOWN']

# -------------------------------
# Preprocessing
# -------------------------------
genre_replacements = {"sci-fi": "science fiction", "rom-com": "romance comedy"}
lemmatizer = WordNetLemmatizer()

def preprocess_genres(genres):
    genres = genres.lower()
    genres = [genre_replacements.get(s.strip(), s.strip()) for s in genres.split(',')]
    return ','.join(genres)

def description_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s<>]', '', text)
    text = re.sub(r'[\d]+','<NUMBER>', text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in stop_words]
    return ' '.join(tokens)

anime_data['cleaned_genres'] = anime_data['genres'].apply(preprocess_genres)
anime_data['cleaned_description'] = anime_data['description'].astype(str).apply(description_preprocess)

movie_data['cleaned_genres'] = movie_data['genres'].apply(preprocess_genres)
movie_data['cleaned_description'] = movie_data['description'].astype(str).apply(description_preprocess)

# -------------------------------
# Combine + Features
# -------------------------------

def genre_tokenizer(x):
    return x.split(',')

anime_data['type'] = 'anime'
movie_data['type'] = 'movie'

movie_data['img_url'] = movie_data['image_url'].apply(
    lambda x: f"http://image.tmdb.org/t/p/w500{x}" if pd.notna(x) and x.strip() != "" else ""
)
anime_data['img_url'] = anime_data['img_url'].fillna("")
combined_data = pd.concat([anime_data, movie_data], ignore_index=True)


genre_vectorizer = CountVectorizer(tokenizer=genre_tokenizer, token_pattern=None)

genre_matrix = genre_vectorizer.fit_transform(combined_data['cleaned_genres'])

vectorizer = TfidfVectorizer(max_features=20000)
description_matrix = vectorizer.fit_transform(combined_data['cleaned_description'])

rating_scaled = MinMaxScaler().fit_transform(combined_data[['rating']])
rating_sparse = sparse.csr_matrix(rating_scaled)

combined_matrix = hstack([genre_matrix, description_matrix, rating_sparse])

# -------------------------------
# Dimensionality Reduction
# -------------------------------
svd = TruncatedSVD(n_components=150, random_state=42)
reduced_matrix = svd.fit_transform(combined_matrix).astype('float32')
faiss.normalize_L2(reduced_matrix)

# -------------------------------
# Build FAISS index
# -------------------------------
d = reduced_matrix.shape[1]
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_INNER_PRODUCT)
index.train(reduced_matrix[:10000])
index.add(reduced_matrix)

# -------------------------------
# Save objects
# -------------------------------
with open("models/combined_data.pkl", "wb") as f:
    pickle.dump(combined_data, f)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("models/svd.pkl", "wb") as f:
    pickle.dump(svd, f)
with open("models/genre_vectorizer.pkl", "wb") as f:
    dill.dump(genre_vectorizer, f)
faiss.write_index(index, "models/faiss.index")

print("Preprocessing complete. Files saved in /models/")
