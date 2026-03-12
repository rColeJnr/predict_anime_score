# ML System Design Doc: Anime Score Predictor

---
### 1.1.1. Why develop the product?
#### Business Goal
With the **Anime Score Predictor (ASP)**, we aim to help anime creators predict how successful their anime will be based on the scores of pre-existing anime.
In the world of anime creation, high production costs mean anime are typically produced by a joint venture of different production studios, known as a *production committee*. When pitching a new anime for development, creators must present the script to this committee, which decides whether to produce the anime.
**ASP** will serve as a tool for both people writing anime scripts and committees deciding whether it is worth investing in the production of the anime.
There are many variables that describe successful anime, such as popularity, fan scores, and fan favorability. While conducting a study of the industry, we found that more often than not, popular anime have a direct correlation with a specific set of genres, themes, and viewership ratings (e.g., PG). This data holds true across different platforms, including *MyAnimeList*, *AniList*, and *AniDB*. Anime in the same genres revolving around similar themes tend to have higher scores and therefore higher popularity, while those outside this set tend to be less popular.
With this knowledge, we are convinced we can develop a platform that predicts the score of any anime based on factors predetermined before production. These factors include, but are not limited to:

- Genres
- Themes
- PG rating
- Synopsis
- Number of episodes
- Episode duration
- Studio
- Source material
- Producer
- And many others

**ASP** aims to aggregate all of this information to predict the score of an anime—or even a specific anime episode.

**ASP** will be presented to users through a web application where they can enter as much information as possible about their anime idea and see the predicted score. After viewing the initial prediction, users can fine-tune parameters until they reach a score that satisfies them, then proceed to make the necessary changes to their script.

Even after a pitch for an anime has been approved, much can change during production. Studios, creators, and writers can continue using **ASP** to track how their changes affect the predicted score. **ASP** can also be useful for already successful anime when writing and developing new seasons or episodes.

With a score predictor that reacts to every change in genres, themes, description, PG rating, number of episodes, etc., **ASP** can prove to be a valuable asset for the entire anime industry.

**Typical user flow of ASP** [Link to schema](https://github.com/rColeJnr/predict_anime_score/blob/fcc08bd89a7e99a69ec8dd70c57297be30f2f573/User%20flow%20graph.png)

**What is the relevance of ML in our project?**
Machine Learning is the core of ASP, while our web app will present a beautiful, user-friendly reactive interface to the user, our ML model will do the heavy lifting 
of the project.

**Process automated** We have no doubt that when writing anime scripts and even during production creators study 
the market of existing anime, comparing their project with many others to see what failed and what worked in the past.
ASP will automate this process by providing a ML service trained on available anime to compare a specific
anime against all the anime with similarities and provide a success probability score.

**Project Success Criteria**
The success of ASP relies on how precisive the ML model prediction of anime scores is. The project is considered
complete when the following technical and operational milestones are met:
### ML Model Milestones
* Model capable of performing sentiment analysis on inputs such as complete anime scripts (scenery, dialogs, etc.).
* Leverage NLP model results with linear analysis of inputs such as number of episodes, PG rating, genres, and others.
* **ASP** yields scores within the predefined failure acceptance range of `0.5`.
    - *Inference Accuracy*: The model achieves a **Mean Absolute Error (MAE) ≤ 0.5**.

### Infrastructure Milestones
* A website through which users can interact with the model.
* The **ASP** website and model process **100 requests per second** with ease.
* Model inference completes in under `300ms`.
* An API that couples the website with the server running the model.

**Project Structure Schema** [Link to schema](https://github.com/rColeJnr/predict_anime_score/blob/bfd8a1320cccc724cca12ab985db1726910e56b9/ML%20Service%20development%20graph.png)

---

### 1.1.2. Task Positioning
To meet our business goal, Anime Score Predictor should:
1. Accept different formats of input: Support categorical data (genres, pg-rating) and unstructured text data (synopsis, script)
2. Sentiment Analysis: Use NLP to evaluate unstructured text data.
3. Provide *"What-if" analysis*: Allow users to modify inputs (e.g. change genres) and see an instant update to the predicted score.
4. Access to the model: Provide a website through which users can query the model.

### MVP vs Final Product

While we intend ASP to leverage as much data as possible in order for it be as precise as possible when predicting scores, for the MVP
we'll keep it as simple as possible, working within the timeframe we have to build and deploy our MVP we will focus on developing a demonstrative tool of what ASP will be able to do.
For simplicity, we will predict the score of an anime based only on: pg-rating, genres and themes. and we will use one data set for training.

### MVP
Our task are:
1. Accept categorical input format (genre, pg-rating and themes).
2. Provide *What-if"* analysis.
3. Provide access to the model.

### Data Engineering && EDA
- We used the [Top Anime Data 2025 dataset](https://www.kaggle.com/datasets/wiltheman/anime-data-set-for-ml).
- After cleaning the data, we produced the `clean_anime_data.csv` dataset [clean_anime_data.ipynb](https://github.com/rColeJnr/predict_anime_score/blob/d7ee1b82d09752ecd5f1fd08c0df47fd254298bd/clean_anime_data.ipynb), available in the repo:
    - Handled multi-label features (Genres/Themes/PG-Rating) via encoding.
    - Merged highly correlated features.
    - Handled missing data and duplicates.
    - Filled missing values.
- We used this dataset for EDA. During EDA, we discovered that genres, themes, and PG rating are major determinants of anime success, so we generated a final `eda_anime_data.csv` dataset [eda_anime_data.ipynb](https://github.com/rColeJnr/predict_anime_score/blob/d7ee1b82d09752ecd5f1fd08c0df47fd254298bd/eda_anime_data.ipynb), available in the repo:
    - EDA revealed that the key categorical features determining anime score are: `genres`, `themes`, and `pg-rating`.
    - EDA also revealed that certain strong genres directly translate to high popularity scores.
    - Outliers were detected, so we used logarithmic normalization to handle them.
    - When comparing score against genres and ratings, we saw that regardless of genre, R+ ratings can massively drop the score of an anime.

### Model training and Deployment
- We split the data 80/20 for training/validation.
- After training different models, **XGBoost** was selected due to the highest `R^2` score.
- We used Pickle to serialize the model.
- **ONNX** was used for optimization, achieving faster inference times.
- The model is served through a **FastAPI** endpoint, accessible via a **Streamlit** webpage.

**Key highlights from training:**
- Compared Random Forest (MAE: 0.434, `R^2`: 0.415), Linear Regression (MAE: 0.439, `R^2`: 0.427), and XGBoost (MAE: 0.407, `R^2`: 0.494).
- XGBoost won; further optimized via GridSearchCV (MAE: 0.401).
- Exported XGBoost model for ONNX optimization.

---

### Baseline (MVP) vs. Advanced (Production) Solution
* **Baseline (MVP):** Standard XGBoost trained on basic metadata (episodes, favorites, rating).
* **Advanced (Production):** A Stacking Ensemble that combines metadata-boosted trees with a Deep Learning layer for textual analysis.

### Validation Strategy: Time-Series Forward Chaining
Standard $k$-fold cross-validation is avoided due to temporal leakage. We utilize **Forward Chaining**:
1. Train on 2020–2025 $\rightarrow$ Test on 2026.
2. Train on 2020–2026 $\rightarrow$ Test on 2027.
This ensures the model learns to predict the *future* based on the *past*.

### Business Rules & Guardrails
* **Rule 1:** "Genre Minimums" — If a user selects 0 genres, the system defaults to an average rated genre for the calculation, same for themes.

---

## Deployment and Monitoring

### Architecture
* **Frontend:** Streamlit-based "Pro Dashboard." hosted on `streamlit community` at https://predictanimescore-rcolejnr.streamlit.app/
* **Inference API:** FastAPI containerized via Docker, hosted on Render at https://anime-score-predictor.onrender.com
* **Model Format:** ONNX for optimized inference.

### Architecture (Production)
To achieve **100 requests per second** and **< 300ms latency**, especially with a model performing "sentiment analysis of complete scripts," we need a robust tech stack.

| Component          | Technology             | Recommendation             | Reason                                                                 |
|--------------------|------------------------|----------------------------|------------------------------------------------------------------------|
| Frontend           | Kotlin multiplatform   | High reactivity for the "fine-tuning" sliders/inputs | Ensures a responsive user interface                                  |
| API Gateway        | Ktor or Go             | Low-overhead handling of concurrent requests | Handles high throughput efficiently                                   |
| ML Inference       | ONNX Runtime           | Drastically faster than standard PyTorch/TensorFlow for production | Optimized for speed in inference tasks                                |
| NLP Engine         | DistilBERT or LightGBM | Full BERT might be too slow for < 300ms on long scripts; distilled models or gradient boosting on text features offer a better speed/accuracy trade-off | Balances speed and accuracy for large scripts                        |

* Implementation of a *Two-Tower Architecture* or a *Multimodal Transformer*, allowing for processing of categorical metadata and text data simultaneously.
* 
### Monitoring (The "Feedback Loop")
The system logs the "Residual Error" (Predicted Score - Actual Score) for every anime. If the Mean Residual exceeds `0.5` over a 30-day window, a "Model Drift" alert is sent to the Data Science team.

---

## Final Product
The final product should:
- Should be trained on as much available data as possible
- Present an ensemble of different models capable of:
  - Predict the success of anime based on a wider range of parameters
  - Provide profound sentiment analysis
  - Provide information explaining the predicted score
  - Provide suggestion on how to change the anime script to improve the score, while staying true to the originality of the data provided.
  - Account for popular anime at the moment and adjust the score prediction accordingly.
  - Account for external factor that can affect anime popularity, such as production studio
- Implement SHAP (SHapley Additive exPlanations) or LIME.
---