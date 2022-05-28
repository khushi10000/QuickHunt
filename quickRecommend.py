# import streamlit as st
# import pickle
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

##content based recommendation on the movies
def app():
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

    ##Vectorising the overview part of the dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    # removing english stop word like a, and , the
    tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    cosin_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    def get_recommendations(title, cosin_sim=cosin_sim):
        idx = row_of_movie_by_title[title]

        sim_scores = list(enumerate(cosin_sim[idx]))
        # sorting of moviesidx based on similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # get top 10 of sorted
        sim_scores = sim_scores[1:31]

        movies_idx = [i[0] for i in sim_scores]
        movies_idx = movies_idx[:5]
        quick_recommend = movies['title'].iloc[movies_idx]
        # Getting the movie name and movie poster
        quick_recommend_movie = []
        quick_recommended_movie_poster = []
        test = []
        for i in quick_recommend:
            quick_recommend_movie.append(i)
        for j in movies_idx:
            quick_movie_id = movies.loc[j]['id']
            quick_recommended_movie_poster.append(poster_image(quick_movie_id))
        return quick_recommend_movie[:5],quick_recommended_movie_poster[:5]
    st.title('Your Favourite Movie Name')
    favourite_movie_name = st.selectbox(
        'Selected Movie',
        movies['title'].values, 89)
    if st.button('Recommend', 16):
        # progress = st.progress(0)
        # for i in range(100):
        #     time.sleep(0.1)
        #     progress.progress(i+1)
        quick_recommended_movie_names,quick_recommended_movie_posters= get_recommendations(favourite_movie_name,cosin_sim=cosin_sim)
        st.balloons()
        st.header("Recommended For You")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.write(quick_recommended_movie_names[0])
            st.image(quick_recommended_movie_posters[0])
        with col2:
            st.write(quick_recommended_movie_names[1])
            st.image(quick_recommended_movie_posters[1])
        with col3:
            st.write(quick_recommended_movie_names[2])
            st.image(quick_recommended_movie_posters[2])
        with col4:
            st.write(quick_recommended_movie_names[3])
            st.image(quick_recommended_movie_posters[3])
        with col5:
            st.write(quick_recommended_movie_names[4])
            st.image(quick_recommended_movie_posters[4])