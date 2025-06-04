# 🧠 Optimal Topics: Streamlit App for Topic Modeling

This app allows users to upload comment data (and optional concern data), generate an interpretable topic model using Gensim LDA, and visualize the results with interactive UMAP clustering.

## 🚀 Features

- 📁 Upload CSV files (comments + optional concerns)
- 📌 Choose which column of text to analyze
- 🧹 Preprocessing with stopword removal
- 🔍 Interactive topic modeling with LDA (via Gensim)
- 📊 Visualization using pyLDAvis
- 🌐 UMAP embedding with HDBSCAN clustering
- 🎨 Interactive Plotly scatter plot with hoverable comments

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the App

```bash
streamlit run topic_umap_app.py
```

## 📁 File Structure

- `topic_umap_app.py` — Main Streamlit application
- `requirements.txt` — Dependencies for Streamlit Cloud or local use

## 💡 Notes

- Clustered comments are labeled (e.g. `Cluster 0`, `Cluster 1`)
- Concerns are displayed but not clustered, labeled as `Concern`
- Topic count is user-adjustable
