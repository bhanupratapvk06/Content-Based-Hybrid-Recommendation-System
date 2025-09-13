import dill
import faiss
import pickle
import pandas as pd
from scipy import sparse
from scipy.sparse import hstack
import streamlit as st

# -------------------------------
# Load objects
# -------------------------------
combined_data = pickle.load(open("models/combined_data.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
genre_vectorizer = dill.load(open("models/genre_vectorizer.pkl", "rb"))
svd = pickle.load(open("models/svd.pkl", "rb"))
index = faiss.read_index("models/faiss.index")

# -------------------------------
# Explainability helpers
# -------------------------------
def get_shared_genres(idx1, idx2):
    g1 = set(combined_data.loc[idx1, 'cleaned_genres'].split(','))
    g2 = set(combined_data.loc[idx2, 'cleaned_genres'].split(','))
    return g1.intersection(g2)

def build_feature_vector(idx, combined_data, genre_vectorizer, vectorizer):
    g_vec = genre_vectorizer.transform([combined_data.loc[idx, "cleaned_genres"]])
    d_vec = vectorizer.transform([combined_data.loc[idx, "cleaned_description"]])
    r_vec = sparse.csr_matrix([[combined_data.loc[idx, "rating"]]])
    return hstack([g_vec, d_vec, r_vec])

def find_cross_recommendations(title, n=10):
    found = combined_data[combined_data['title'].str.contains(title, case=False, regex=False)]
    if found.empty:
        return f"No matches for '{title}'"
    
    idx = found.index[0]
    input_type = combined_data.loc[idx, 'type']
    target_type = 'movie' if input_type == 'anime' else 'anime'
    
    vec = build_feature_vector(idx, combined_data, genre_vectorizer, vectorizer)
    vec_reduced = svd.transform(vec).astype("float32")
    
    distances, indices = index.search(vec_reduced, len(combined_data))
    
    recs = []
    for i, rec_idx in enumerate(indices[0]):
        if combined_data.loc[rec_idx, 'type'] != target_type:
            continue
        
        shared_genres = get_shared_genres(idx, rec_idx)
        explanation = ""
        if shared_genres:
            explanation += f"Shares themes: {', '.join(shared_genres)}. "
        if abs(combined_data.loc[idx, 'rating'] - combined_data.loc[rec_idx, 'rating']) < 0.5:
            explanation += "Similar audience rating. "
        
        recs.append({
            "title": combined_data.loc[rec_idx, "title"],
            "rating": combined_data.loc[rec_idx, "rating"],
            "description": combined_data.loc[rec_idx, "description"],
            "genres": combined_data.loc[rec_idx, "genres"],
            "image": combined_data.loc[rec_idx, "img_url"],
            "similarity": float(distances[0][i]),
            "explanation": explanation.strip()
        })
        if len(recs) >= n:
            break
    
    return pd.DataFrame(recs)



# -------------------------------
# Streamlit
# -------------------------------

st.set_page_config(page_title="Anime & Movie Recommender", page_icon="🎬", layout="wide")


st.markdown("""
    <style>
    body {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #1f6feb;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #1f6feb;
        padding: 0.5em;
        background-color: #161b22;
        color: #c9d1d9;
    }
    .recommendation-card {
        background-color: #161b22;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.6);
        margin-bottom: 15px;
    }
    h1, h3, p {
        color: #c9d1d9;
    }
    .highlight {
        color: #58a6ff;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    "<h1 style='text-align:center; color:#58a6ff;'>🎬 Anime & Movie Recommendation System</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#8b949e;'>Find similar movies and anime based on genres, ratings, and more ⚡</p>",
    unsafe_allow_html=True
)

# User input
user_input = st.text_input("🔎 Enter a movie or anime title:")


if st.button("✨ Recommend"):
    if user_input:
        results = find_cross_recommendations(user_input, n=12)

        if isinstance(results, str):
            st.warning(results)
        else:
            st.markdown("## Top Recommendations")

            num_columns = 1
            for i in range(0, len(results), num_columns):
                cols = st.columns(num_columns)
                for j, col in enumerate(cols):
                    if i + j < len(results):
                        row = results.iloc[i + j]
                        with col:
                            image_url = row['image']
                            if pd.isna(image_url) or not isinstance(image_url, str) or image_url.strip() == "":
                                image_url = "https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg"

                            genre_html = ""
                            if row['genres']:
                                genres = [g.strip() for g in row['genres'].split(',')]
                                genre_html = " ".join([
                                    f"<span style='background-color:#1f6feb; color:white; padding:3px 8px; border-radius:5px; margin:2px;'>{g}</span>"
                                    for g in genres
                                ])
                            
                            st.markdown(f"""
                                <div class="recommendation-card" style="display:flex; align-items:flex-start; gap:20px; min-height:250px; padding:15px; border:1px solid #eee; border-radius:10px;">
                                    <div style="flex-shrink:0; width:200px; height:250px; overflow:hidden; border-radius:8px;">
                                        <img src="{image_url}" style="width:100%; height:100%; object-fit:cover;">
                                    </div>
                                    <div style="flex-grow:1; text-align:left;">
                                        <h4 class="highlight" style="margin-top:10px;">{row['title']} <span style = "color:#1f6feb">--- Similarity: ({row['similarity']:.2f})</span></h3>
                                        <p>⭐ <strong>{row['rating']}</strong></p>
                                        <p>{genre_html}</p>
                                        <p> {row['description']}</p>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
