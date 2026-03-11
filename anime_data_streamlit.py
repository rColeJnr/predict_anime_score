
import streamlit as st
import requests

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

API_URL = "no deployment url yet"

st.title("Anime Score Predictor")
st.write("Predict the My Anime List score of your favorite anime")

with st.form('prediction_form'):
    col = st.columns(1)
    with col:
        rating = st.selectbox("Rating:", ["PG", "PG-13", "R", "R+", "Rx"])

    genre_selection = st.multiselect("Genres:", top_genres)
    theme_selection = st.multiselect("Themes:", top_themes)

    submit = st.form_submit_button("Predict Score")

if submit:
    payload = {
        'genres': genre_selection,
        'themes': theme_selection,
        'rating': rating
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()['predicted_score']
        st.metric(label="Predicted Score", value=f"{result:.3f}")
        if result >= 8.0:
            st.success('This sounds like a good anime, but is it better than Jujutsu Kaisen?')
        elif result >= 6.5:
            st.info("Not bad, but Gojou Satoru wouldn't waste his time watching this.")
        else:
            st.info("The only acceptable anime here is: Город в котором меня нет.")
    else:
        st.error("Failed to get prediction from the server.")