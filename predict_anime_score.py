import numpy as np
import pandas as pd
import pickle

with open('anime_data_model.pkl', 'rb') as file:
    model = pickle.load(file)

expected_features = [
    'log_episodes', 'log_favorites', 'mystery', 'suspense', 'sports', 'drama', 
    'slice_of_life', 'romance', 'adventure', 'supernatural', 'gourmet', 'action', 
    'fantasy', 'comedy', 'sci-fi', 'theme_iyashikei', 'theme_childcare', 
    'theme_gag_humor', 'theme_organized_crime', 'theme_adult_cast', 
    'theme_visual_arts', 'theme_love_status_quo', 'theme_love_polygon', 
    'theme_performing_arts', 'theme_combat_sports', 'theme_urban_fantasy', 
    'theme_cgdct', 'theme_team_sports', 'theme_otaku_culture', 'theme_showbiz', 
    'theme_historical', 'theme_time_travel', 'theme_racing', 'theme_delinquents', 
    'theme_detective', 'theme_medical', 'theme_military', 'theme_samurai', 
    'theme_school', 'rating_PG', 'rating_PG-13', 'rating_R', 'rating_R+', 'rating_Rx'
]

def predict_anime_score(genre_list, theme_list, rating_suffix, expected_features, model):
    input_data = {col: 0 for col in expected_features}

    for genre in genre_list:
        if genre in input_data:
            input_data[genre] = 1

    for theme in theme_list:
        if theme in input_data:
            input_data[theme] = 1

    rating_col = f"rating_{rating_suffix}"
    if rating_col in input_data:
        input_data[rating_col] = 1
    else:
        pass

    input_df = pd.DataFrame([list(input_data.values())], columns=expected_features)
    return model.predict(input_df)[0]

# score_jjk = predict_anime_score(
#     47,
#     130619,
#     ['action', 'supernatural', 'fantasy'],
#     ['theme_school', 'theme_urban_fantasy'],
#     'R',
#     expected_features=expected_features,
#     model=model
# )

# print(f"JJK Predicted Score: {score_jjk:.2f}") 
# scored predicted to be 8.3, actual score 8.5 👍🏾