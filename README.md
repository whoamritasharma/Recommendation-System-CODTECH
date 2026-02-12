# Recommendation-System-CODTECH
# Recommendation System with Collaborative Filtering



## 📋 Project Overview

This project implements a comprehensive **Recommendation System** using **Collaborative Filtering** and **Matrix Factorization** techniques. The system is trained on the **MovieLens 100K dataset** containing 100,000 movie ratings and provides personalized movie recommendations to users.

**Internship:** CODTECH IT Solutions  
**Task:** Task 4 - Recommendation System  
**Objective:** Build a recommendation system showcasing recommendation results and evaluation metrics  

---

## 🎯 Objectives

- Implement multiple recommendation algorithms
- Compare User-Based and Item-Based Collaborative Filtering
- Apply Matrix Factorization techniques (SVD, NMF)
- Evaluate models using RMSE and MAE metrics
- Generate personalized recommendations for users
- Analyze recommendation quality and diversity

---

## 🛠️ Technologies Used

### Programming Language
- **Python 3.8+**

### Libraries & Frameworks
- **Surprise:** Recommendation system library
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Evaluation:** scikit-surprise

---

## 📂 Project Structure

```
recommendation-system/
│
├── recommendation_system.ipynb    # Main Jupyter notebook
├── README.md                      # Project documentation (this file)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file
├── SETUP.md                       # Setup instructions
│
└── results/                       # (Generated during execution)
    ├── model_comparison.png       # Performance comparison charts
    ├── predictions_analysis.png   # Prediction error analysis
    └── recommendations_samples.txt # Sample recommendations
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/recommendation-system.git
cd recommendation-system
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook recommendation_system.ipynb
```

---

## 📊 Dataset

### MovieLens 100K Dataset
- **Source:** GroupLens Research
- **Size:** 100,000 ratings
- **Users:** 943 users
- **Items:** 1,682 movies
- **Rating Scale:** 1-5 stars
- **Sparsity:** ~93.7% (very sparse)
- **Time Period:** September 1997 - April 1998

The dataset is automatically downloaded by the Surprise library on first run.

**Dataset Statistics:**
- Average rating: ~3.5
- Ratings per user: ~106
- Ratings per movie: ~60
- Most active user: 737 ratings
- Most rated movie: 583 ratings

---

## 🔍 Recommendation Algorithms

### 1. **User-Based Collaborative Filtering**
- Finds similar users based on rating patterns
- Recommends items liked by similar users
- Similarity metric: Cosine similarity
- K-nearest neighbors: k=40

**Pros:**
- Serendipitous recommendations
- Good for new items

**Cons:**
- Scalability issues with many users
- Cold start problem for new users

---

### 2. **Item-Based Collaborative Filtering**
- Finds similar items based on user ratings
- Recommends items similar to those user liked
- Similarity metric: Cosine similarity
- K-nearest neighbors: k=40

**Pros:**
- More stable than user-based
- Better scalability
- Explainable recommendations

**Cons:**
- Less diverse recommendations
- Cold start for new items

---

### 3. **SVD (Singular Value Decomposition)**
- Matrix factorization technique
- Decomposes user-item matrix into latent factors
- Parameters:
  - Factors: 100
  - Epochs: 20
  - Learning rate: 0.005
  - Regularization: 0.02

**Advantages:**
- Handles sparsity well
- Captures latent features
- Better accuracy
- Dimensionality reduction

---

### 4. **NMF (Non-Negative Matrix Factorization)**
- Constraint: All factors are non-negative
- Interpretable latent features
- Parameters:
  - Factors: 15
  - Epochs: 50

**Advantages:**
- Interpretable factors
- Parts-based representation
- Non-negative constraints

---

## 📈 Model Performance

### Evaluation Metrics

| Model | RMSE | MAE | Description |
|-------|------|-----|-------------|
| **User-Based CF** | ~0.97 | ~0.76 | Baseline collaborative filtering |
| **Item-Based CF** | ~0.95 | ~0.74 | Better than user-based |
| **SVD** | **~0.93** | **~0.73** | **Best performer** |
| **NMF** | ~0.96 | ~0.75 | Good interpretability |

**RMSE (Root Mean Squared Error):** Lower is better - measures prediction accuracy  
**MAE (Mean Absolute Error):** Lower is better - average absolute deviation  

### Cross-Validation Results (SVD)
- **5-Fold CV RMSE:** 0.93 ± 0.01
- **5-Fold CV MAE:** 0.73 ± 0.01

---

## 💡 Key Features

✅ **Multiple Algorithms**
- User-based and item-based collaborative filtering
- Matrix factorization (SVD, NMF)
- Comprehensive comparison

✅ **Robust Evaluation**
- Train-test split (75-25)
- Cross-validation
- Multiple metrics (RMSE, MAE)

✅ **Rich Visualizations**
- Rating distributions
- Model performance comparisons
- Prediction error analysis
- Recommendation diversity charts

✅ **Personalized Recommendations**
- Top-N recommendations for each user
- Interactive recommendation function
- Explanation of recommendations

✅ **Analysis & Insights**
- Data sparsity analysis
- Recommendation coverage
- Diversity metrics
- Popular items analysis

---

## 📝 Usage Example

### Generate Recommendations for a User

```python
from surprise import SVD, Dataset
from surprise.model_selection import train_test_split

# Load data
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25)

# Train model
svd = SVD(n_factors=100, n_epochs=20)
svd.fit(trainset)

# Get recommendations for user '196'
user_id = '196'
# Get all items not yet rated by user
all_items = df['item_id'].unique()
rated_items = df[df['user_id'] == user_id]['item_id'].unique()
items_to_predict = [i for i in all_items if i not in rated_items]

# Predict ratings
predictions = []
for item in items_to_predict:
    pred = svd.predict(user_id, item)
    predictions.append((item, pred.est))

# Sort and get top 10
predictions.sort(key=lambda x: x[1], reverse=True)
top_10 = predictions[:10]

# Display
for i, (movie_id, rating) in enumerate(top_10, 1):
    print(f"{i}. Movie {movie_id} - Predicted Rating: {rating:.2f}")
```

---

## 🎨 Visualizations

The notebook includes:

1. **Data Exploration**
   - Rating distribution histogram
   - Ratings per user/item distributions
   - Top rated movies chart
   - Most popular movies chart

2. **Model Performance**
   - RMSE comparison bar chart
   - MAE comparison bar chart
   - Cross-validation fold performance

3. **Prediction Analysis**
   - Prediction error distribution
   - Actual vs Predicted scatter plot
   - Residual plot

4. **Recommendation Analysis**
   - Most frequently recommended items
   - Recommendation diversity metrics
   - Coverage analysis

---

## 🔮 Future Enhancements

- [ ] Implement Deep Learning models (Neural Collaborative Filtering)
- [ ] Add content-based filtering using movie metadata
- [ ] Create hybrid recommendation system
- [ ] Build web interface using Flask/Streamlit
- [ ] Add real-time recommendation updates
- [ ] Implement A/B testing framework
- [ ] Add explainable AI features
- [ ] Scale to larger datasets (MovieLens 1M, 20M)
- [ ] Deploy as REST API
- [ ] Add user feedback loop

---

## 📚 Learning Outcomes

Through this project, I learned:

✔️ Collaborative filtering algorithms (user-based, item-based)  
✔️ Matrix factorization techniques (SVD, NMF)  
✔️ Recommendation system evaluation metrics  
✔️ Handling sparse data matrices  
✔️ Cold start problem and solutions  
✔️ Recommendation diversity and coverage  
✔️ Cross-validation for recommendation systems  
✔️ Practical implementation using Surprise library  

---

## 🎓 Key Concepts

### Collaborative Filtering
- **Core Idea:** "Users who agreed in the past will agree in the future"
- **Types:** User-based, Item-based
- **Similarity Metrics:** Cosine, Pearson correlation

### Matrix Factorization
- **Core Idea:** Decompose user-item matrix into latent factors
- **Techniques:** SVD, NMF, ALS
- **Benefits:** Handles sparsity, captures hidden patterns

### Evaluation Metrics
- **RMSE:** Penalizes large errors more
- **MAE:** Average absolute error
- **Precision@K:** Accuracy of top-K recommendations
- **Recall@K:** Coverage of relevant items in top-K

### Cold Start Problem
- **New User:** No rating history
- **New Item:** No ratings yet
- **Solutions:** Hybrid methods, content-based filtering, popularity-based

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- **CODTECH IT Solutions** for the internship opportunity
- **GroupLens Research** for the MovieLens dataset
- **Surprise library** developers
- Recommendation systems research community

---

## 📞 Contact

For any queries or feedback, please reach out:

- **Email:** whoamritasharma@gmail.com

---

## 📖 References

- [Surprise Documentation](https://surprise.readthedocs.io/)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Matrix Factorization Techniques](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- [Recommender Systems Handbook](https://www.springer.com/gp/book/9780387858203)

---

## ⭐ Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

## 📊 Project Stats

- **Lines of Code:** ~1,500+
- **Models Implemented:** 4
- **Evaluation Metrics:** 2 (RMSE, MAE)
- **Visualizations:** 15+
- **Dataset Size:** 100,000 ratings

---

**Last Updated:** February 2026  
**Status:** ✅ Completed
