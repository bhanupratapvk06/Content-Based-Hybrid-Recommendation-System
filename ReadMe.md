---
title: "Anime & Movie Recommender"
colorFrom: "blue"
colorTo: "purple"
sdk: gradio
sdk_version: "3.44"
app_file: app.py
pinned: false
---

# 🎨 Anime & Movie Recommendation System

A **content-based recommendation system** that suggests similar anime
and movies using genres, plot descriptions, and ratings.  
It leverages **TF-IDF**, **FAISS (Facebook AI Similarity Search)**, and
**dimensionality reduction (TruncatedSVD)** for efficient similarity
search, with an interactive **Gradio UI** for exploration.

------------------------------------------------------------------------

## 📌 Features

- Hybrid feature engineering with **genres**, **descriptions**, and **ratings**.
- **Content preprocessing** with tokenization, lemmatization, and stopword removal.
- **Dimensionality reduction** using TruncatedSVD for efficient similarity search.
- **FAISS-based nearest neighbor search** for fast recommendations.
- **Maximal Marginal Relevance (MMR)** for diverse recommendation results.
- User-friendly **web interface** with recommendation explanations (shared genres, similar ratings, etc.).

------------------------------------------------------------------------

## 🔧 Project Structure

```
├── app.py
├── preprocess_and_build.py
├── Data/
│   ├── anime-dataset-2023.csv
│   ├── TMDB_movie_dataset_v11.csv
├── models/
│   ├── combined_data.pkl
│   ├── vectorizer.pkl
│   ├── genre_vectorizer.pkl
│   ├── reduced_matrix.pkl
│   ├── svd.pkl
│   ├── faiss.index
```

------------------------------------------------------------------------

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/bhanupratapvk06/Content-Based-Hybrid-Recommendation-System.git
cd anime-movie-recommender
```

2. Create a virtual environment & install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

3. Download datasets and place them under the `Data/` directory:

- `anime-dataset-2023.csv`
- `TMDB_movie_dataset_v11.csv`

------------------------------------------------------------------------

## 🚀 Usage

### Step 1: Preprocess data and build models

```bash
python preprocess_and_build.py
```

### Step 2: Run the Gradio app

```bash
python app.py
```

Open the local URL (usually `http://localhost:8501/`) in your browser.

------------------------------------------------------------------------

## 📊 Example

1. Enter an anime or movie title (e.g., *Your Name*).  
2. The system finds **cross-domain recommendations** (anime → movies or movies → anime).  
3. Results include similarity score, genres, ratings, and short explanations.

------------------------------------------------------------------------

## 📦 Requirements

- Python 3.8+
- pandas, numpy, scipy
- scikit-learn
- nltk
- faiss
- dill
- gradio

Install all with:

```bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📛 License

This project is licensed under the MIT License -- feel free to use and modify.