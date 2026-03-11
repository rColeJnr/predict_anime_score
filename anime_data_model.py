import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('anime_data_eda.csv')

top_genres = [
    'mystery', 'suspense', 'sports', 'drama', 'slice_of_life', 
    'romance', 'adventure', 'supernatural', 'gourmet', 'action', 
    'fantasy', 'comedy', 'sci-fi'
]
top_themes = [
    'theme_iyashikei', 'theme_childcare', 'theme_gag_humor', 'theme_organized_crime', 
    'theme_adult_cast', 'theme_visual_arts', 'theme_love_status_quo', 'theme_love_polygon', 
    'theme_performing_arts', 'theme_combat_sports', 'theme_urban_fantasy', 'theme_cgdct', 
    'theme_team_sports', 'theme_otaku_culture', 'theme_showbiz', 'theme_historical', 
    'theme_time_travel', 'theme_racing', 'theme_delinquents', 'theme_detective', 
    'theme_medical', 'theme_military', 'theme_samurai', 'theme_school'
]
metrics = ['score', 'popularity', 'episodes', 'favorites']

df['log_favorites'] = np.log1p(df['favorites'])

rating_dummies = pd.get_dummies(df['rating'], prefix = 'rating', drop_first = True)
df_model = pd.concat([df, rating_dummies], axis = 1)


X = pd.concat([
    df[['log_episodes', 'log_favorites']], 
    df[top_genres], 
    df[top_themes],
    rating_dummies
], axis=1)
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Ready to train on {X_train.shape[0]} samples with {X_train.shape[1]} features.")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mae = np.mean(abs(y_pred - y_test))
r2 = r2_score(y_test, y_pred)

print(f"Random Forest MAE: {mae:.3f}") #  0.434
print(f"Random Forest R^2: {r2:.3f}") # 0.415

# Now we compare this model with LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

mae_lr = np.mean(abs(y_pred_lr - y_test))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression MAE: {mae_lr:.3f}") # 0.439
print(f"Linear Regression R^2: {r2_lr:.3f}") # 0.427

# Now we compare this model with XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=6, 
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = np.mean(abs(y_pred_xgb - y_test))
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost MAE: {mae_xgb:.3f}") # 0.407
print(f"XGBoost R^2: {r2_xgb:.3f}") # 0.494

# XGBoost is our clear winner, with 0.494

from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}

xgb_model_reg = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')

grid_search = GridSearchCV(
    estimator=xgb_model_reg,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test) # 0.401
print(f"Optimized MAE: {mean_absolute_error(y_test, y_pred_best):.3f}")
print(f"Optimized R^2: {r2_score(y_test, y_pred_best):.3f}") # 0.5

# Extracting the weights
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Weight': lr_model.coef_
}).sort_values(by='Weight', ascending=False)

print("Top 10 Positive Score Drivers:")
print(coeff_df.head(10))

#                   Feature    Weight
# 17        theme_gag_humor  0.391081
# 15        theme_iyashikei  0.378900
# 30       theme_historical  0.330326
# 32           theme_racing  0.309204
# 27      theme_team_sports  0.278285
# 23  theme_performing_arts  0.242942
# 24    theme_combat_sports  0.235292
# 16        theme_childcare  0.233028
# 1           log_favorites  0.225150
# 19       theme_adult_cast  0.169933
print("\nTop 10 Negative Score Drivers:")
print(coeff_df.tail(10))
#                   Feature    Weight
# 28    theme_otaku_culture -0.059155
# 12                fantasy -0.060344
# 18  theme_organized_crime -0.071831
# 3                suspense -0.077567
# 0            log_episodes -0.089422
# 40           rating_PG-13 -0.128646
# 37          theme_samurai -0.129852
# 41               rating_R -0.238096
# 43              rating_Rx -0.380664
# 42              rating_R+ -0.546340

import matplotlib.pyplot as plt
import seaborn as sns

residuals = y_test - y_pred_xgb

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.axvline(0, color='red', linestyle='--')
plt.title('XGBoost Residuals (Errors) Distribution')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.show()

# saving model on device

import pickle
with open('anime_data_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("Model successfully saved")

# Testing model

with open('anime_data_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

sample_anime = X_test.iloc[[0]] 
prediction = loaded_model.predict(sample_anime)

print(f"Predicted Score: {prediction[0]:.2f}") # 6 .13
print(f"Actual Score: {y_test.iloc[0]:.2f}") # 6.11