import os
import dill
import gdown
import faiss
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# -------------------------------
# Setup model directory
# -------------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# Model files and Google Drive IDs
# -------------------------------
files_to_download = {
    "combined_data.pkl": "1odp6rHp4R3fgIVKl8REK_kVcbrP0We51",
    "vectorizer.pkl": "1ngxRoIJ40Srnq0F2elNiB3ATwHoD2jL6",
    "genre_vectorizer.pkl": "1EM1RwPRX4obB1tSc_Fd0sJtvjnwb8qJd",
    "reduced_matrix.pkl": "1c1A26pb47WIKpxuOz-vrOMsN1OPgvuLy",
    "svd.pkl": "14MhVhKdEaPg5qOXhl_LyAOfGLTYEci0u",
    "faiss.index": "1JEGTS31JLXc_SEXyiW8svOH61pYv5hyH"
}

# Download missing files
for filename, file_id in files_to_download.items():
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)

# Load models
with open(os.path.join(MODEL_DIR, "combined_data.pkl"), "rb") as f:
    combined_data = pickle.load(f)

with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(MODEL_DIR, "genre_vectorizer.pkl"), "rb") as f:
    genre_vectorizer = dill.load(f)

with open(os.path.join(MODEL_DIR, "reduced_matrix.pkl"), "rb") as f:
    reduced_matrix = dill.load(f)
    reduced_matrix = reduced_matrix.astype('float32')

with open(os.path.join(MODEL_DIR, "svd.pkl"), "rb") as f:
    svd = pickle.load(f)

faiss_index_path = os.path.join(MODEL_DIR, "faiss.index")
index = faiss.read_index(faiss_index_path)

# -------------------------------
# Helper functions
# -------------------------------
def get_shared_genres(idx1, idx2):
    g1 = set(combined_data.loc[idx1, 'cleaned_genres'].split(','))
    g2 = set(combined_data.loc[idx2, 'cleaned_genres'].split(','))
    return g1.intersection(g2)

def maximal_marginal_relevance(query_vec, candidate_vecs, candidate_indices, top_n=10, diversity=0.5):
    sim_to_query = cosine_similarity(candidate_vecs, query_vec.reshape(1, -1))
    sim_between = cosine_similarity(candidate_vecs)
    selected = []
    while len(selected) < top_n and len(selected) < len(candidate_indices):
        if not selected:
            idx = np.argmax(sim_to_query)
            selected.append(idx)
        else:
            remaining = list(set(range(len(candidate_indices))) - set(selected))
            mmr_scores = []
            for i in remaining:
                relevance = sim_to_query[i]
                diversity_penalty = max(sim_between[i][j] for j in selected)
                mmr_scores.append(diversity * relevance - (1 - diversity) * diversity_penalty)
            idx = remaining[np.argmax(mmr_scores)]
            selected.append(idx)
    return [candidate_indices[i] for i in selected]

def find_cross_recommendations(title, n=12, diversity=0.6):
    found = combined_data[combined_data['title'].str.contains(title, case=False, regex=False)]
    if found.empty:
        return f"No matches for '{title}'"
    
    idx = found.index[0]
    input_type = combined_data.loc[idx, 'type']
    target_type = 'movie' if input_type == 'anime' else 'anime'
    vec = reduced_matrix[idx].reshape(1, -1).astype('float32')
    distances, indices = index.search(vec, 50)
    candidate_indices = [i for i in indices[0][1:] if combined_data.loc[i, 'type'] == target_type]
    if not candidate_indices:
        return f"No recommendations found for '{title}' in target type '{target_type}'"
    candidate_vecs = reduced_matrix[candidate_indices].astype('float32')
    diverse_indices = maximal_marginal_relevance(vec, candidate_vecs, candidate_indices, top_n=n, diversity=diversity)

    recs = []
    for rec_idx in diverse_indices:
        shared_genres = get_shared_genres(idx, rec_idx)
        explanation = ""
        if shared_genres:
            explanation += f"Shares themes: {', '.join(shared_genres)}. "
        if abs(combined_data.loc[idx, 'rating'] - combined_data.loc[rec_idx, 'rating']) < 0.5:
            explanation += "Similar audience rating. "
        image_url = combined_data.loc[rec_idx, "img_url"]
        if pd.isna(image_url) or not isinstance(image_url, str) or image_url.strip() == "":
            image_url = "https://upload.wikimedia.org/wikipedia/commons/1/14/No_Image_Available.jpg"
        recs.append({
            "Title": combined_data.loc[rec_idx, "title"],
            "Rating": combined_data.loc[rec_idx, "rating"],
            "Genres": combined_data.loc[rec_idx, "genres"],
            "Description": combined_data.loc[rec_idx, "description"],
            "Image": image_url,
            "Explanation": explanation.strip(),
        })
    return recs

# -------------------------------
# Gradio interface
# -------------------------------

def recommend_gradio(title):
    recs = find_cross_recommendations(title)
    
    if isinstance(recs, str):
        return recs
    
    html_output = ""
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        
        genre_html = ""
        if rec.get("Genres"):
            genres = [g.strip() for g in rec["Genres"].split(',')]
            genre_html = " ".join([
                f"<span class='genre-badge'>{g}</span>"
                for g in genres
            ])
        
        html_output += f"""
        <div class="rec-card">
            <img src="{rec.get('Image', '')}">
            <div class="rec-content">
                <h4>{rec.get('Title', 'Unknown')} — ⭐ {rec.get('Rating', 'N/A')}</h4>
                <p>{genre_html}</p>
                <p class="desc">{rec.get('Description', '')}</p>
                <p class="explanation"><i>{rec.get('Explanation', '')}</i></p>
            </div>
        </div>
        """
    return html_output or "<p style='color:#c9d1d9;'>No recommendations found.</p>"

# -------------------------------
# Launch Gradio Blocks UI
# -------------------------------

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <style>
    #search-bar textarea {
        width: 100% !important;
        font-size: 1.2em !important;
        padding: 12px !important;
        border-radius: 12px !important;
        background-color: #161b22 !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
    }
    .rec-card {
        display: flex;
        gap: 20px;
        margin-bottom: 20px;
        background-color: #0d1117;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .rec-card img {
        width: 150px;
        height: 220px;
        object-fit: cover;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .rec-content h4 {
        color: #58a6ff;
        margin-bottom: 6px;
        font-family: 'Segoe UI', sans-serif;
    }
    .rec-content .desc {
        color: #c9d1d9;
        font-size: 0.95em;
    }
    .rec-content .explanation {
        color: #8b949e;
        font-size: 0.85em;
    }
    .genre-badge {
        background-color: #1f6feb;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        margin: 2px;
        font-size: 0.85em;
    }
    </style>
    """)

    gr.Markdown("<h1 style='color:#58a6ff; text-align:center;'>🎬 Anime & Movie Recommendation System</h1>")
    gr.Markdown("<p style='color:#c9d1d9; text-align:center;'>Find similar movies and anime based on genres, ratings, and more ⚡</p>")
    
    search_input = gr.Textbox(
        label="",
        placeholder="Type a movie or anime title here...",
        elem_id="search-bar",
        show_label=False
    )
    search_button = gr.Button("Search", variant="primary")
    

    output_html = gr.HTML()
    

    search_button.click(fn=recommend_gradio, inputs=search_input, outputs=output_html)

demo.launch()
