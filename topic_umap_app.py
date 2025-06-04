import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from gensim import corpora, models
import importlib
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import tempfile
import umap
import hdbscan
import plotly.express as px
import plotly.graph_objs as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re


# -------------------------
# Helper Functions
# -------------------------
def preprocess_text(texts):
    return [
        [word for word in re.sub(r'[^a-zA-Z ]', '', doc).lower().split()
         if len(word) > 2 and word not in ENGLISH_STOP_WORDS]
        for doc in texts
    ]

# -------------------------
# App UI
# -------------------------
st.title("Topic Modeling and UMAP Visualizer")

uploaded_comments = st.file_uploader("Upload Comments CSV", type="csv")
uploaded_concerns = st.file_uploader("Upload Optional Concerns CSV", type="csv")

if uploaded_comments:
    progress = st.progress(0)
    status = st.empty()

    df_comments = pd.read_csv(uploaded_comments)
    comment_col = st.selectbox("Select column from Comments file", df_comments.columns)

    if uploaded_concerns:
        df_concerns = pd.read_csv(uploaded_concerns)
        concern_col = st.selectbox("Select column from Concerns file", df_concerns.columns)
    else:
        df_concerns = pd.DataFrame()
        concern_col = None

    # -------------------------
    # Preprocessing
    # -------------------------
    status.info("📋 Preprocessing text data...")
    comments_texts = df_comments[comment_col].dropna().astype(str).tolist()
    concern_texts = df_concerns[concern_col].dropna().astype(str).tolist() if concern_col else []

    comments_texts = list(dict.fromkeys([c.lower() for c in comments_texts]))
    concern_texts = list(dict.fromkeys([c.lower() for c in concern_texts]))
    all_texts = comments_texts + concern_texts

    processed_texts = preprocess_text(all_texts)
    progress.progress(20)
    status.success("✅ Text preprocessing complete")

    # -------------------------
    # Topic Modeling
    # -------------------------
    num_topics = st.slider("Select number of topics for LDA", min_value=2, max_value=20, value=5)
    status.info(f"🔍 Building LDA topic model with {num_topics} topics...")

    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    st.write(f"📊 Number of documents: {len(processed_texts)}")
    st.write(f"🧾 Number of unique tokens: {len(dictionary)}")
    st.write("🧠 Training LDA model...")

    with st.spinner("Running Gensim LDA model..."):
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,
            eval_every=None
        )

    progress.progress(40)
    status.success("✅ LDA topic modeling complete")

    # Show top words per topic
    st.subheader("📌 Top Words Per Topic")
    for i, topic in lda_model.show_topics(num_topics=num_topics, formatted=False):
        st.markdown(f"**Topic {i+1}:** " + ", ".join([word for word, prob in topic]))

    # -------------------------
    # LDA Visualization
    # -------------------------
    st.subheader("LDA Topic Model Visualization")
    with st.spinner("Preparing interactive LDA visualization..."):
        vis = gensimvis.prepare(lda_model, corpus, dictionary)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
            pyLDAvis.save_html(vis, tmpfile.name)
            st.components.v1.html(open(tmpfile.name, 'r').read(), height=1000, scrolling=True, width=1600)
    progress.progress(60)

    # -------------------------
    # UMAP + HDBSCAN
    # -------------------------
    st.subheader("UMAP + HDBSCAN Visualization")
    status.info("🔧 Vectorizing and running UMAP...")

    text_data = comments_texts + concern_texts
    vectorizer = CountVectorizer(min_df=5, stop_words='english')
    vectors = vectorizer.fit_transform(text_data).toarray()

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding = reducer.fit_transform(vectors)

    progress.progress(75)
    status.success("✅ UMAP embedding completed")

    status.info("🔍 Running HDBSCAN clustering on comments...")
    clusters = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(vectors[:len(comments_texts)])
    st.write(f"🔹 Number of clusters found (excluding noise): {len(set(clusters)) - (1 if -1 in clusters else 0)}")

    # Label clusters as 'Cluster X' or 'Noise', and concerns as 'Concern'
    cluster_labels = [
        f"Cluster {c}" if c != -1 else "Noise"
        for c in clusters
    ] + ['Concern'] * len(concern_texts)

    progress.progress(85)

    # -------------------------
    # UMAP Visualization
    # -------------------------
    status.info("📊 Creating interactive UMAP scatter plot...")
    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    umap_df['Label'] = cluster_labels
    umap_df['Text'] = text_data

    fig = px.scatter(
        umap_df,
        x='UMAP1',
        y='UMAP2',
        color='Label',
        title='UMAP Clustering with HDBSCAN',
        hover_name='Text'
    )
    fig.update_traces(marker=dict(size=8), hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"))
    fig.update_layout(hovermode='closest', showlegend=True)
    fig.update_layout(width=900)
    st.plotly_chart(fig, use_container_width=False, config={"responsive": True})
    progress.progress(100)
    status.success("🎉 All steps complete!")