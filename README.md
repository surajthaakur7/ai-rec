# 🛍️ AI Product Recommendation System

A hybrid AI-powered product recommendation system built using 
Collaborative Filtering, Content-Based Filtering, and Google Gemini LLM.
Developed as an institute project by a team of 3.

---

## 👥 Team
- Suraj Thakur
- Sarbjot Singh  
- Krish Sharma

---

## 📌 About the Project
This system recommends Amazon products to users based on their past 
behavior and product similarity. Users can also chat naturally with 
an AI assistant to get product recommendations.

### How it works:
1. **Collaborative Filtering** — finds similar users and recommends 
   products they liked
2. **Content Based Filtering** — finds similar products using 
   TF-IDF on review text
3. **Hybrid Model** — combines both for better accuracy
4. **Gemini LLM** — converts results into natural conversation

---

## 📂 Project Structure

```
ai-rec/
├── Notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   ├── 03_content_based.ipynb
│   ├── 04_hybrid_model.ipynb
│   └── 05_evaluation.ipynb
├── Datasets/
│   └── amazon_cleaned.csv
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset
Amazon Food Reviews dataset from Kaggle.
Contains 5,68,454 rows with UserId, ProductId, Score, Summary and Text.

> ⚠️ Full dataset (Ratings.csv) is excluded from repo due to size.
> Download from: [https://www.kaggle.com/code/qwikfix/amazon-recommendation-dataset/input]
> After downloading place it in the Datasets/ folder.

---

## 🛠️ Tech Stack
- Python
- Jupyter Notebook
- Scikit-learn (KNN, TF-IDF, Cosine Similarity)
- Google Gemini API (gemini-2.5-flash)
- Streamlit
- Pandas, NumPy, Matplotlib, Seaborn

---

## 📈 Model Results

| Model | MAE | MSE | R² |
|-------|-----|-----|----|
| Collaborative Filtering | 2.51 | 8.88 | -5.0 |
| Content Based Filtering | 0.83 | 1.45 | -0.45 |
| Hybrid Model | 0.87 | 1.15 | +0.14 |

Hybrid model achieved the best overall performance with 
a positive R² score.

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/surajthaakur7/ai-rec.git
cd ai-rec
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key
Create a `.env` file in the project root:
GEMINI_API_KEY=your_api_key_here

### 4. Run the app
```bash
streamlit run app.py
```

---

## 💬 Features
- 🤖 Chat naturally with AI about any product
- 🛍️ Get personalized recommendations by User ID
- ⭐ See real ratings and reviews from dataset
- 🔍 Search products by description or category

---

## 🚧 Future Improvements
- Add product images
- Deploy on cloud (Streamlit Cloud or Hugging Face)
- Improve CF with more user rating data
- Add more product categories

---
