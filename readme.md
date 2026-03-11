# 🎬 Anime Score Predictor

A very simple Machine Learning project that predicts the user score (0.0 - 10.0) of an anime based on my anime list score. The dataset used for this project can be found at Kaggle - Top Anime Data 2025.



## 🚀 Project Overview
Score prediction is achieved throught the input of data, the same features used to train the model. though user inputed data (number of episodes, genres, themes, rating) the model provides a prediction of the MyAnimeList (MAL) user score.

### Key Performance Metrics:
| Metric | Baseline (Linear Reg) | Final Model (XGBoost) |
| :--- | :--- | :--- |
| **MAE (Mean Absolute Error)** | 0.439 | **0.401** |
| **R² (Explained Variance)** | 0.427 | **0.500** |

---

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Core Libraries:** Pandas, NumPy, Scikit-Learn
* **Model:** XGBoost (Extreme Gradient Boosting)
* **Optimization:** GridSearchCV (Hyperparameter Tuning)
* **Deployment:** Streamlit

---

## 📂 Repository Structure
* `clean_anime_data.ipynb`: data cleaning workflow.
* `eda_anime_data.ipynb`: eda workflow.
* `anime_data_model.py`: model teaching workflow.
* `predict_anime_score.py`: The Streamlit web application.
* `anime_data_model.pkl`: The serialized, trained XGBoost model.
* `system_design_doc.md`: Technical documentation and ML system design.

---

## 🔧 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rColeJnr/predict_anime_score.git

2. **Install dependencies**

3. **Run the web app**
    ```bash
    streamlit run predict_anime_score.py
