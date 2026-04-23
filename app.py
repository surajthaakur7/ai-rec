import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

# ─── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="AI Product Recommender",
    page_icon="🛍️",
    layout="wide"
)

# ─── Load and Prepare Data ─────────────────────────────
@st.cache_resource
def load_models():
    df = pd.read_csv('Datasets/Reviews.csv')
    df['Summary'] = df['Summary'].fillna('')
    df['Text'] = df['Text'].fillna('')
    df['combined_text'] = df['Summary'] + ' ' + df['Text']

    # CF setup — only store matrix, no full similarity
    user_counts = df['UserId'].value_counts()
    df_cf = df[df['UserId'].isin(user_counts[user_counts >= 10].index)]

    user_product_matrix = df_cf.pivot_table(
        index='UserId',
        columns='ProductId',
        values='Score'
    ).fillna(0)

    # CBF setup — only store tfidf matrix, no full similarity
    product_reviews = df.groupby('ProductId')['combined_text'].apply(
        lambda x: ' '.join(x)
    ).reset_index()

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(product_reviews['combined_text'])

    avg_scores = df.groupby('ProductId')['Score'].mean()

    return df, user_product_matrix, product_reviews, tfidf, tfidf_matrix, avg_scores

# ─── Hybrid Recommend ──────────────────────────────────
def hybrid_recommend(user_id, df, user_product_matrix, tfidf_matrix, product_reviews, avg_scores, alpha=0.3, n=5):

    # CF — compute similarity only for this user
    if user_id in user_product_matrix.index:
        user_vec = user_product_matrix.loc[[user_id]].values
        user_sim = cosine_similarity(user_vec, user_product_matrix)[0]

        mean_ratings = user_product_matrix.mean(axis=1).values
        ratings_diff = user_product_matrix.sub(
            user_product_matrix.mean(axis=1), axis=0
        ).values

        predicted = mean_ratings[user_product_matrix.index.get_loc(user_id)] + (
            user_sim.dot(ratings_diff) /
            (np.abs(user_sim).sum() + 1e-9)
        )

        cf_scores = pd.Series(predicted, index=user_product_matrix.columns)

        rated = user_product_matrix.loc[user_id]
        rated_products = rated[rated > 0].index.tolist()
        cf_scores = cf_scores.drop(
            [p for p in rated_products if p in cf_scores.index]
        )
    else:
        cf_scores = pd.Series(dtype=float)

    # CBF — compute similarity only for user's products
    user_products = df[df['UserId'] == user_id]['ProductId'].tolist()
    cbf_scores = pd.Series(dtype=float)

    for product_id in user_products[:5]:
        if product_id in product_reviews['ProductId'].values:
            idx = product_reviews[
                product_reviews['ProductId'] == product_id
            ].index[0]
            product_vec = tfidf_matrix[idx]
            sim_scores = cosine_similarity(product_vec, tfidf_matrix).flatten()
            sim_series = pd.Series(sim_scores, index=product_reviews['ProductId'])
            sim_series = sim_series.drop(product_id)
            cbf_scores = pd.concat([cbf_scores, sim_series])

    cbf_scores = cbf_scores.groupby(cbf_scores.index).mean()

    # Combine CF + CBF
    all_products = set(cf_scores.index).union(set(cbf_scores.index))
    hybrid_scores = {}
    for product in all_products:
        cf_val = cf_scores.get(product, 0)
        cbf_val = cbf_scores.get(product, 0)
        hybrid_scores[product] = (alpha * cf_val) + ((1 - alpha) * cbf_val)

    return pd.Series(hybrid_scores).nlargest(n)

# ─── Search Dataset ────────────────────────────────────
def search_dataset(query, tfidf, tfidf_matrix, product_reviews, df, avg_scores):
    query_vec = tfidf.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-5:][::-1]

    results = []
    for idx in top_indices:
        product_id = product_reviews.iloc[idx]['ProductId']
        avg = round(avg_scores.get(product_id, 0), 2)
        review = df[df['ProductId'] == product_id]['Text'].iloc[0][:200]
        results.append(
            f"Product: {product_id} | Avg Rating: {avg}/5 | Review: {review}"
        )

    return "\n".join(results)

# ─── Ask Gemini ────────────────────────────────────────
def ask_gemini(user_message, user_id, df, user_product_matrix, tfidf, tfidf_matrix, product_reviews, avg_scores):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    recommendations_text = ""
    if user_id and user_id in df['UserId'].values:
        recs = hybrid_recommend(
            user_id, df, user_product_matrix,
            tfidf_matrix, product_reviews, avg_scores
        )
        rec_details = []
        for pid, score in recs.items():
            avg = round(avg_scores.get(pid, 0), 2)
            review = df[df['ProductId'] == pid]['Summary'].iloc[0] if len(df[df['ProductId'] == pid]) > 0 else ""

            rec_details.append(
    f"- Product ID: {pid} | "
    f"Avg Rating: {avg}/5 | "
    f"Summary: {review} | "
    f"Review: {df[df['ProductId'] == pid]['Text'].iloc[0][:100]}"
)
            
        recommendations_text = "Personalized recommendations:\n" + "\n".join(rec_details)

    dataset_context = search_dataset(
        user_message, tfidf, tfidf_matrix, product_reviews, df, avg_scores
    )

    prompt = f"""You are a helpful AI product recommendation assistant for an Amazon product dataset.

{recommendations_text}

Relevant products from dataset:
{dataset_context}

User asks: {user_message}

Answer naturally and helpfully. Always mention the Product ID, average rating and a brief description for each product you recommend. Keep it concise and friendly."""

    # Try models in order if one fails
    models_to_try = ["gemini-2.5-flash"]
    #, "gemini-1.5-flash", "gemini-1.5-flash-8b"]
    
    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                continue  # try next model
            else:
                raise e  # different error, raise it

    return "⚠️ All models are currently busy. Please try again in a moment!"
# ─── UI ────────────────────────────────────────────────
st.title("🛍️ AI Product Recommendation System")
st.markdown("Powered by Hybrid CF + CBF models and Google Gemini")
st.divider()

# Load models
with st.spinner("Loading AI models... please wait ⏳"):
    df, user_product_matrix, product_reviews, tfidf, tfidf_matrix, avg_scores = load_models()

st.success("✅ Models loaded and ready!")

# Two columns layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("👤 Your Profile")
    user_id = st.text_input(
        "Enter User ID (optional)",
        placeholder="e.g. A3SGXH7AUHU8GW"
    )

    st.markdown("**💡 Sample User IDs:**")
    for uid in df['UserId'].value_counts().head(5).index:
        st.code(uid)

    if user_id and user_id in df['UserId'].values:
        st.success("✅ User found!")
        user_ratings = df[df['UserId'] == user_id]
        st.metric("Products Rated", len(user_ratings))
        st.metric("Avg Rating Given", round(user_ratings['Score'].mean(), 2))
    elif user_id:
        st.error("❌ User ID not found in dataset")

with col2:
    st.subheader("💬 Chat with AI")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask anything about products..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_gemini(
                    prompt, user_id, df, user_product_matrix,
                    tfidf, tfidf_matrix, product_reviews, avg_scores
                )
                st.write(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })