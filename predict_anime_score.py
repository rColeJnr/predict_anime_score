import numpy as np
import pandas as pd
import pickle

with open('anime_data_model.pkl', 'rb') as file:
    model = pickle.load(file)

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

features = ['log_episodes', 'log_favorites'] + top_genres + top_themes + [
    'PG - Children', 
    'PG-13 - Teens 13 or older', 
    'R - 17+ (violence & profanity)',
    'R+ - Mild Nudity',
    'Rx - Hentai'
]

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

def predict_anime_score(episodes, favorites, genre_list, theme_list, rating_suffix, expected_features, model):
    input_data = {col: 0 for col in expected_features}

    input_data['log_episodes'] = np.log1p(episodes)
    input_data['log_favorites'] = np.log1p(favorites)

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

import streamlit as st

st.title("Anime Score Predictor")
st.write("Predict the My Anime List score of your favorite anime")

with st.form('prediction_form'):
    col1, col2 = st.columns(2)
    with col1:
        eps = st.number_input("Episodes", min_value=1, value=12)
        favs = st.number_input("Favorites", min_value=0, value=5000)
    with col2:
        rating = st.selectbox("Rating:", ["PG", "PG-13", "R", "R+", "Rx"])

    genre_selection = st.multiselect("Genres:", top_genres)
    theme_selection = st.multiselect("Themes:", top_themes)

    submit = st.form_submit_button("Predict Score")

if submit:
    result = predict_anime_score(
        eps,
        favs,
        genre_selection,
        theme_selection,
        rating,
        expected_features=expected_features,
        model=model
    )
    st.metric(label="Predicted Score", value=f"{result:.3f}")

    if result >= 8.0:
        st.success('This sounds like a good anime, but is it better than Jujutsu Kaisen?')
    elif result >= 6.5:
        st.info("Not bad, but Gojou Satoru wouldn't waste his time watching this.")
    else:
        st.info("The only acceptable anime here is: Город в котором меня нет.")