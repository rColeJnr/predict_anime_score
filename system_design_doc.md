# ML System Design Doc: Anime Score Predictor

## 1. Goals and Prerequisites

### 1.1. Why develop the product?
* **Business Goal:** The Anime Prediction Score (APS) is a decision-support system for production committees and studios. It aims to facilitate the "Green-light" process on anime project ideas, identifying the expected critical success and market reception of an anime concept before high-cost production begins.
* **What is the relevance of our ML?** Traditional green-lighting relies on many factors. Anime production are funded not by one studio, but by a production committee (a group of studios), so we want for this ML system to be something anime creators can look for after they have created their stories and before turning to anime production committeess simple load our ml project and predict whether their anime idea is worth trying at all.

### 1.2. Business Requirements and Constraints
* **Unified Analytics Suite:** The system must provide a "What-If" simulation tool where users can swap genres, studios, and themes to see real-time score fluctuations. a tool like this would help shape the story of the anime into something that will probably achieve a higher score.
* **Cold-Start Capability:** The model must provide accurate predictions for "Original" anime (no existing Manga/Light Novel data) using thematic similarity mapping.
* **Interpretability (SHAP):** Every prediction must be accompanied by a feature-importance breakdown (e.g., "The 'Cyberpunk' theme is currently trending, adding +0.3 to your score").
* **Constraint:** All inference must occur in $< 300ms$ to support interactive web-based brainstorming.

---

## 2. Methodology

### 2.1. Task Positioning
Technical Task: **Supervised Multi-modal Regression**. The system uses structured metadata (categorical/numerical) and unstructured text (synopsis sentiment) to predict a continuous target variable (`score`).
Currently the system only supports numerical prediction, but ASP should be able to predict the score of an anime based on the provided title and synopsis. A title like HunterXHunter is surely easier to remember and spread around, in contrast to title like "Город в котором меня нет".

### 2.2. Solution Architecture


* **Data Aggregator:** Automated scrapers for MyAnimeList and AniList.
* **Feature Store:** Centralized versioning of "Genre Saturation" and "Studio Prestige" scores.
* **Model Engine:** An ensemble of XGBoost and CatBoost models.
* **Observation Layer:** Monitoring for "Concept Drift" (e.g., when a genre like 'Isekai' begins to see diminishing returns in scores).

### 2.3. Feature Engineering & Assumptions
| Entity | Attribute Name | Source | Processing / Engineering |
| :--- | :--- | :--- | :--- |
| **Hype** | `log_favorites` | MAL API | Log-normalization to handle "Super-hit" outliers. |
| **Sentiment** | `synopsis_vec` | NLP Module | BERT Embeddings to capture "Dark" vs. "Wholesome" tones. |
| **Tempo** | `seasonal_trend` | Internal | Cyclical encoding (Sine/Cosine) of release months. |
| **Categorical** | `studio_tier` | Internal | Rank-encoding based on the historical mean score of the studio. |

---

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