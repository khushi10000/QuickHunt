
##importing the modules
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import requests
import time


##loading the dataset (csv) file in form of pkl
movies = pickle.load(open('movies_3.pkl', 'rb'))
movies = pd.DataFrame(movies)
movieId_data = pickle.load(open('movie_id_3.pkl', 'rb'))
movieId_data = pd.DataFrame(movieId_data)
id_row_relation = pickle.load(open('index_map_3.pkl', 'rb'))
id_row_relation = pd.DataFrame(id_row_relation)
row_of_movie_by_title = pickle.load(open('index_of_movies_3.pkl', 'rb'))
ratings_by_users = pickle.load(open('ratings_3.pkl', 'rb'))
ratings_by_users = pd.DataFrame(ratings_by_users)


##calculating indexes for calculating top movies
mean_of_votes = movies['vote_average'].mean()
minimum_of_votes = movies['vote_count'].quantile(0.90)
segregated_movies = movies.copy().loc[movies['vote_count'] >= minimum_of_votes]


def rating(x, minimum_of_votes=minimum_of_votes, mean_of_votes=mean_of_votes):
    voters = x['vote_count']
    avg_vote = x['vote_average']
    return (voters / (voters + minimum_of_votes) * avg_vote) + (minimum_of_votes / (minimum_of_votes + voters)) * mean_of_votes

def poster_image(movieId):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
        "Accept-Encoding": "*",
        'accept': 'application/json',
        "Connection": "keep-alive"
    }
    response = requests.get(
        'https://api.themoviedb.org/3/movie/{}?api_key=6e4b7f49bdf8472167640728bc8a48cc&language=en-US'.format(
            movieId), headers=headers)
    path = response.json()
    temp = "https://image.tmdb.org/t/p/original/" + path['poster_path']
    response.close()

    return temp
def app():
    st.header('Explore More')
    segregated_movies['score'] = segregated_movies.apply(rating, axis=1)
    segregated_movies.sort_values('score', ascending=False)
    global_top_movies = segregated_movies.head(20)
    global_top_movies = segregated_movies['title'].head(10)
    top_recommend_movie = []
    top_movie_poster = []
    for i in global_top_movies:
        top_recommend_movie.append(i)
        top_movie_poster.append(poster_image(movieId_data.loc[i]['id']))

    col6, col7, col8, col9, col10 = st.columns(5)
    with col6:
        st.write(top_recommend_movie[0])
        st.image(top_movie_poster[0])
    with col7:
        st.write(top_recommend_movie[1])
        st.image(top_movie_poster[1])
    with col8:
        st.write(top_recommend_movie[2])
        st.image(top_movie_poster[2])
    with col9:
        st.write(top_recommend_movie[3])
        st.image(top_movie_poster[3])
    with col10:
        st.write(top_recommend_movie[4])
        st.image(top_movie_poster[4])

    col16, col17, col18, col19, col110 = st.columns(5)
    with col16:
        st.write(top_recommend_movie[5])
        st.image(top_movie_poster[5])
    with col17:
        st.write(top_recommend_movie[6])
        st.image(top_movie_poster[6])
    with col18:
        st.write(top_recommend_movie[7])
        st.image(top_movie_poster[7])
    with col19:
        st.write(top_recommend_movie[8])
        st.image(top_movie_poster[8])
    with col110:
        st.write(top_recommend_movie[9])
        st.image(top_movie_poster[9])
    st.balloons()
