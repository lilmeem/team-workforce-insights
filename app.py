"""
Career Compass - job recommendation dashboard.

Loads precomputed skill vectors from 06_recommendation_engine.ipynb
(recommendation_artifacts.npz + postings_clean.parquet) and lets a user
pick skills to get ranked job recommendations plus a skill-gap list.

Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

BASE = "./"


@st.cache_resource
def load_artifacts():
    artifacts = np.load(f"{BASE}recommendation_artifacts.npz", allow_pickle=True)
    postings = pd.read_parquet(
        f"{BASE}postings_clean.parquet",
        columns=["job_id", "title", "company_name", "location"],
    )
    weighted_matrix = artifacts["weighted_matrix"]
    idf = artifacts["idf"]
    matrix_norms = artifacts["matrix_norms"]
    skill_list = list(artifacts["skill_list"])
    skill_to_idx = {s: i for i, s in enumerate(skill_list)}
    return postings, weighted_matrix, idf, matrix_norms, skill_list, skill_to_idx


def recommend(user_skills, weighted_matrix, idf, matrix_norms, skill_to_idx, n_skills, top_n=10):
    user_vec = np.zeros(n_skills, dtype=np.float32)
    for s in user_skills:
        if s in skill_to_idx:
            user_vec[skill_to_idx[s]] = 1.0
    user_weighted = user_vec * idf
    user_norm = np.linalg.norm(user_weighted)

    scores = weighted_matrix @ user_weighted
    denom = matrix_norms * user_norm
    denom[denom == 0] = 1
    cosine = scores / denom

    top_idx = np.argsort(-cosine)[:top_n]
    return top_idx, cosine[top_idx]


def skill_gaps(user_skills, top_idx, weighted_matrix, idf, skill_list, n_gaps=8):
    # Recover the binary skill matrix from the weighted one (weighted = binary * idf, idf > 0 everywhere)
    binary_sub = (weighted_matrix[top_idx] > 0).astype(np.float32)
    user_set = set(user_skills)
    freq = binary_sub.sum(axis=0)
    gaps = [(skill_list[i], int(freq[i])) for i in range(len(skill_list))
            if skill_list[i] not in user_set and freq[i] > 0]
    gaps.sort(key=lambda x: -x[1])
    return gaps[:n_gaps]


st.set_page_config(page_title="Career Compass", layout="wide")
st.title("Career Compass")
st.caption(
    "Pick your skills to see the most relevant job postings and the "
    "in-demand skills you're missing. Recommendations are based on a "
    "content-based match against skills extracted from 123,849 LinkedIn "
    "job postings - see the repo's notebooks for methodology."
)

postings, weighted_matrix, idf, matrix_norms, skill_list, skill_to_idx = load_artifacts()
n_skills = len(skill_list)

selected_skills = st.multiselect(
    "Your skills",
    options=skill_list,
    default=["Python", "SQL", "Excel"] if "Python" in skill_list else skill_list[:3],
)
top_n = st.slider("Number of recommendations", min_value=5, max_value=25, value=10)

if not selected_skills:
    st.info("Pick at least one skill above to see recommendations.")
else:
    top_idx, scores = recommend(
        selected_skills, weighted_matrix, idf, matrix_norms, skill_to_idx, n_skills, top_n=top_n
    )
    results = postings.iloc[top_idx][["title", "company_name", "location"]].copy()
    results["match"] = (scores * 100).round(1).astype(str) + "%"

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Recommended postings")
        st.dataframe(results.reset_index(drop=True), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Skills you're missing")
        gaps = skill_gaps(selected_skills, top_idx, weighted_matrix, idf, skill_list)
        if gaps:
            gap_df = pd.DataFrame(gaps, columns=["skill", "postings among your top matches"])
            st.dataframe(gap_df, use_container_width=True, hide_index=True)
        else:
            st.write("No common gaps found among your top matches - nice fit!")

st.divider()
st.caption(
    "v1 gazetteer covers 133 skills - a skill you enter that isn't in the list above "
    "isn't recognized yet. See the project README for known limitations."
)
