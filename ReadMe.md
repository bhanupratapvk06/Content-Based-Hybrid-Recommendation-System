# 🎬 Anime & Movie Hybrid Recommendation System

A **content-based hybrid recommendation system** that suggests **similar movies for a given anime** and **similar anime for a given movie**.  
The system uses **TF-IDF, genre embeddings, ratings, dimensionality reduction (SVD), and FAISS indexing** for efficient similarity search, and is wrapped in an interactive **Streamlit web app**.

---

## ✨ Features
- Cross-domain recommendations (Anime ↔ Movies)
- Content-based hybrid features:
  - Genres (bag-of-words)
  - Descriptions (TF-IDF)
  - Ratings (normalized)
- Dimensionality reduction with **Truncated SVD**
- Fast similarity search using **FAISS**
- Clean and modern **Streamlit UI** with explanations:
  - Shared genres
  - Similar audience rating
- Image previews for recommendations

---

## 🛠 Tech Stack
- **Python** (pandas, numpy, scikit-learn, scipy)
- **Natural Language Processing**: NLTK, TF-IDF
- **Dimensionality Reduction**: TruncatedSVD
- **Similarity Search**: FAISS
- **Web App**: Streamlit
- **Persistence**: dill, pickle

---

## 📂 Project Structure
```
├── Data/                        # Raw datasets
│   ├── anime-dataset-2023.csv
│   └── TMDB_movie_dataset_v11.csv
│
├── models/                      # Saved preprocessing + FAISS index
│   ├── combined_data.pkl
│   ├── vectorizer.pkl
│   ├── genre_vectorizer.pkl
│   ├── svd.pkl
│   └── faiss.index
│
├── preprocess_and_build.py       # Preprocess datasets & build FAISS index
├── app.py                        # Streamlit web app for recommendations
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/anime-movie-recommender.git
cd anime-movie-recommender
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK resources
The preprocessing script requires **stopwords** and **WordNet lemmatizer**:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🚀 Usage

### Step 1: Preprocess and build index
Run this once to generate `models/` artifacts.
```bash
python preprocess_and_build.py
```

### Step 2: Launch the app
```bash
streamlit run app.py
```

### Step 3: Open in browser
Navigate to:
```
http://localhost:8501
```

Enter a movie or anime title to get **cross-recommendations** 🎉

---

## 📊 Example
- Input: *Your Name* (Anime)
- Output: Recommendations of **movies** with similar genres/themes/ratings.  

---

## 🔮 Future Improvements
- Add collaborative filtering to enrich hybrid model
- Improve genre extraction with embeddings
- Deploy on **Streamlit Cloud / Hugging Face Spaces**
- Add user history & personalization

---

## 🤝 Contributing
Contributions are welcome!  
Feel free to fork, submit issues, or open a pull request.

---

## 📜 License
MIT License © 2025 Bhanu Pratap Singh
