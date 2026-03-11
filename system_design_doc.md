# ML System Design Doc: Anime Score Predictor

### 1.1.1. Why develop the product?
**Business Goal:** With *Anime Score Predictor (APS)* we aim to help anime creators predict
how successful their anime will be based on the scores of pre-existing anime.
In the world of anime creation, due to high production costs usually anime a produced by a joint
of different production studios *a production committee*.
So when pitching a new anime for development the creators must present the script to a committee,
and the committee decides on whether to produce the anime or not.

*Anime Score Prediction* will serve as a tool, both for people writing an anime script 
as well as committees deciding on whether it's worth to invest money on the production of
the anime.

There is a lot of variables descriptive of successful anime, such as popularity, fan scores, 
fan favoritism and others, while conducting a study of the industry we found out that more often than
not popular anime have a direct correlation with a set number of genres, themes and
viewership rating *PG*. And this data holds true across different platforms *My Anime List, AniList, AniDB, etc.*
Same genres of anime revolving around the same themes tend to have higher scores and therefor higher popularity, 
and anime that don't belong to this set of genres and themes tend to be less popular.
With this knowledge we were convinced we could develop a platform that would predict the score of any anime 
based on factors pre-determined before the production of the anime, these factors include, but not limited by:
- genres, themes, pg-rating, synopsis, number of episodes, episode duration, studio, source material, producer and many others

**Anime Score Predictor** aims to agglomerate all of this info and predict the score of an anime or even a specific anime episode.

ASP would be presented to users through a web application where the user would enter as much information as
possible regarding his idea for an anime and see the predicted score, after seeing the initial score prediction,
if necessary the user can fine-tune the parameters until he reaches a score that pleases him, and then proceed
to make the necessary changes to his script.
Even after a pitch for an anime has been approved, during production a lot can change, studios, creators and writers
can continue to use ASP to keep track of how their changes affect the predicted score.
ASP can also be useful for already successful anime when writing and developing a new season or episode.

With a score predictor that reacts to every change to genres, themes, description, pg-rating, number of episodes, etc.,
ASP can prove to be a valuable asset for whole anime industry.

**Typical user flow of ASP**
**ADD LINK HERE**

**What is the relevance of ML in our project?** Machine Learning is the core of ASP, while our website app
will present a beautiful, user-friendly reactive interface to the user, our ML model will do the heavy lifting 
of the project.

**Process automated** We have no doubt that when writing anime scripts and even during production creators study 
the market of existing anime, comparing their project with many others to see what failed and what worked in the past.
ASP will automate this process by providing a ML service trained on available anime to compare a specific
anime against all the anime with similarities and provide a success probability score.

**Project Success Criteria**
The success of ASP relies on how precisive the ML model prediction of anime scores is. The project is considered
complete when the following technical and operational milestones are met:
* Model capable of doing sentiment analysis of inputs such as complete anime script (scenery, dialogs and so on...).
* Leverage the NLP model result with linear analysis of inputs such as: Number of episodes, pg-rating, genres and others.
* Anime Score Predictor should yield scores within the predefined failure acceptance range `o.3`.
  * Inference Accuracy: The model achieves a `Mean Absolute Error smaller or equal to 0.3`.

A website through which users can interact with our model
* The ASP website and our model should be able to process 100 requests per second with easy.
* Model inference must occur in $< 300ms$.
* An API that couples together the website and the server running the model.

**LINK TO DIAGRAMS**
Technical Task: **Supervised Multi-modal Regression**. The system uses structured metadata (categorical/numerical) and unstructured text (synopsis sentiment) to predict a continuous target variable (`score`).
Currently, the system only supports numerical prediction, but ASP should be able to predict the score of an anime based on the provided title and synopsis. A title like HunterXHunter is surely easier to remember and spread around, in contrast to title like "Город в котором меня нет".

### 1.1.2. Task Positioning
To meet our business goal, Anime Score Predictor should:
1. Accept different formats of input: Support categorical data (genres, pg-rating) and unstructured text data (synopsis, script)
2. Sentiment Analysis: Use NLP to evaluate unstructured text data.
2. Provide "What-if" analysis: Allow users to modify inputs (e.g. change genres) and see an instant update to the predicted score.
3. Access to the model: Provide a website through which users can query the model.

## MVP vs Final Product

While we intend ASP to leverage as much data as possible in order for it be as precise as possible when predicting scores, for the MVP
we'll keep it as simple as possible, working within the timeframe we have to build and deploy our MVP we will focus on developing a demonstrative tool of what ASP will be able to do.
For simplicity, we will predict the score of an anime based only on: pg-rating, genres and themes. and we will use one data set for training.

#### MVP
Our task are:
1. Accept categorical input format (genre, pg-rating and themes).
2. Provide "What-if" analysis.
3. Access to the model.

### Data Engineering && EDA
- We use the Top Anime Data 2025 dataset https://www.kaggle.com/datasets/wiltheman/anime-data-set-for-ml
- after cleaning the data we produce the `clean_anime_data.csv` dataset [link to clean_anime_data.ipynb] available in the same repo
    - Handled multi-label features (Genres/Themes/PG-Rating) via encoding
    - Merged highly correlated features into one
    - Handled missing data and duplicates
    - Filled missing data.
- we use this dataset for EDA, during EDA we discover that genres, themes and pg-rating are major determiners of 
anime success, so we generate a final `eda_anime_data.csv` dataset [link to eda_anime_data.ipynb] available in the same repo
  - EDA revealed that the key categorical features determining anime score are: `genres`, `themes` and `pg-rating`
  - EDA also revealed that a strong genre that exactly translate to a high score in popularity.
  - Outliers were detected so we used logarithmic normalization of data to handle them.
  - WHen compared score against genre and ratings we saw our regardless of genre the rating of +18 can massively drop the score of an anime

### Model training and Deployment
- When training the model we split the data by 80/20
- After training in different model XGBoost was selected because it presented the highest R^2 score `0.494`
- We use Pickle to extract the model
- ONNX is used for optimization. achieving faster inference times.
- We serve the model through an API available through a simple webpage we built using streamlit 

## 3. Model Development & Validation

### 3.1. Baseline vs. Advanced Solution
* **Baseline (MVP):** Standard XGBoost trained on basic metadata (episodes, favorites, rating).
* **Advanced (Production):** A Stacking Ensemble that combines metadata-boosted trees with a Deep Learning layer for textual synopsis analysis.

### 3.2. Validation Strategy: Time-Series Forward Chaining
Standard $k$-fold cross-validation is avoided due to temporal leakage. We utilize **Forward Chaining**:
1. Train on 2020–2025 $\rightarrow$ Test on 2026.
2. Train on 2020–2026 $\rightarrow$ Test on 2027.
   This ensures the model learns to predict the *future* based on the *past*.

---

## 4. Inference and Optimization

### 4.1. Performance & Hardware
* **Quantization:** Models are pruned and quantized to minimize the `.pkl` size for serverless deployment (Streamlit/AWS Lambda).
  Monthly* **Trigger Logic:** * **Scheduled:** Monthly retraining on newly completed seasonal titles.

### 4.2. Business Rules & Guardrails
* **Rule 1:** "Genre Minimums" — If a user selects 0 genres, the system defaults to an average rated genre for the calculation.
* **Rule 2:** "Confidence Intervals" — Predictions with low feature support (very rare genre combos) must display a "High Uncertainty" warning.
* **Rule 3:** The system will never suggest "Removing" a genre to increase a score if that genre is core to the synopsis (NLP verification).

---

## 5. Deployment and Monitoring

### 5.1. Architecture
* **Frontend:** Streamlit-based "Pro Dashboard."
* **Inference API:** FastAPI containerized via Docker.

### 5.2. Monitoring (The "Feedback Loop")
The system logs the "Residual Error" (Predicted Score - Actual Score) for every anime. If the Mean Residual exceeds 0.5 over a 30-day window, a "Model Drift" alert is sent to the Data Science team.

---

#### Final Product
The final product should:
- should be trained on as much available data as possible
- present an ensemble of different models capable of:
  - Predict the success of anime based on a wider range of parameters
  - Provide profound sentiment analysis
  - Provide information explaining the predicted score
  - Provide suggestion on how to change the anime script to improve the score, while staying true to the originality of the data provided.
  - Account for popular anime at the moment and adjust the score prediction accordingly.
  - Account for external factor that can affect anime popularity, such as production studio

---