#importing the modules
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import requests
#getting images for the poster from api
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
        # response.close()
        # st.write(temp)
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

        return movies['title'].iloc[movies_idx]

    # st.title(get_recommendations('Total Eclipse', cosin_sim=cosin_sim))

    ##loading and traing the svd for Collaborative Recommendation
    reader = Reader()
    data = Dataset.load_from_df(ratings_by_users[['userId', 'movieId', 'rating']], reader)
    svd = SVD()
    cross_validate(svd, data, measures=['RMSE'], cv=5, verbose=True)
    train = data.build_full_trainset()
    svd.fit(train)

    # st.title(svd.predict(1, 302))

    def recommendation(user_id, title):
        index = row_of_movie_by_title[title]

        # content based
        # getting cosin matrix row of the particular movie
        similarity_of_selected_with_other_movies = list(enumerate(cosin_sim[int(index)]))
        # Sorting the movie similarity keeping the row number of that in mind so using index to sort as the similarity between them
        similarity_of_selected_with_other_movies = sorted(similarity_of_selected_with_other_movies, key=lambda x: x[1],
                                                          reverse=True)
        # getting the top 30 moviies
        similarity_of_selected_with_other_movies = similarity_of_selected_with_other_movies[1:30]
        # getting the row number of that movie to show the title
        similar_movie_index = [i[0] for i in similarity_of_selected_with_other_movies]

        # Getting the row of top 30 sorted movie from movies dataset
        selected_movie_details = movies.iloc[similar_movie_index][['title', 'vote_count', 'vote_average', 'id']]
        # Getting The id of the Selected movies according to the content based recommendation
        selected_movie_details = selected_movie_details[selected_movie_details['id'].isin(movieId_data['id'])]

        # CF
        # Predicting the ratings_by_users that the user will give to the movies selected by content based recommendation
        selected_movie_details['est'] = selected_movie_details['id'].apply(
            lambda x: svd.predict(user_id, id_row_relation.loc[x]['movieId']).est)
        # Now sorting the ratings_by_users in descending order to get top values
        selected_movie_details = selected_movie_details.sort_values('est', ascending=False)
        # Getting the yop 5 movies
        selected_movie_details = selected_movie_details[:5]

        # Getting the movie name and movie poster
        hybrid_recommend_movie = []
        recommended_movie_poster = []
        for i in selected_movie_details['title']:
            hybrid_recommend_movie.append(i)
            recommended_movie_poster.append(poster_image(movieId_data.loc[i]['id']))
        return hybrid_recommend_movie[:5], recommended_movie_poster[:5]

    meanvote = movies['vote_average'].mean()
    minimumvote = movies['vote_count'].quantile(0.90)
    final_movies = movies.copy().loc[movies['vote_count'] >= minimumvote]

    def rating(x, minimumvote=minimumvote, meanvote=meanvote):
        voters = x['vote_count']
        avg_vote = x['vote_average']
        return (voters / (voters + minimumvote) * avg_vote) + (minimumvote / (minimumvote + voters)) * meanvote

    # st.title(recommend_for(200,'Species'))
    # User select the favourite movie so the content based recommendation system can calculate similar movies to it
    st.title('Let Us Know About You')
    st.title('Your Favourite Movie Name')
    favourite_movie_name = st.selectbox(
        'Selected Movie',
        movies['title'].values, 1)

    # Knowing More About The User Activities to estimate ratings_by_users the user will give tio the movies
    st.title('Select A Movie Name')
    movie_name_1 = st.selectbox(
        'Selected Movie',
        movies['title'].values, 2)
    rating_for_movie_1 = st.slider('x', 0, 5, key="1")  # 👈 this is a widget
    st.title('Select A Movie Name')
    movie_name_2 = st.selectbox(
        'Selected Movie',
        movies['title'].values, 3)
    rating_for_movie_2 = st.slider('x', 0, 5, key="2")  # 👈 this is a widget
    st.title('Select A Movie Name')
    movie_name_3 = st.selectbox(
        'Selected Movie',
        movies['title'].values, 4)
    rating_for_movie_3 = st.slider('x', 0, 5, key="3")  # 👈 this is a widget

    ##Main Recommendation
    if st.button('Recommend', 2):
        all_user_id = ratings_by_users['userId']
        current_user = all_user_id.max()
        x = row_of_movie_by_title[movie_name_1]
        current_movie_id_1 = int(movies.loc[x]['id'])
        ratings_by_users = ratings_by_users.append(
            {'userId': current_user, 'movieId': current_movie_id_1, 'rating': rating_for_movie_1,
             'timestamp': 1111111111}, ignore_index=True)
        x = row_of_movie_by_title[movie_name_2]
        current_movie_id_2 = int(movies.loc[x]['id'])
        ratings_by_users = ratings_by_users.append(
            {'userId': current_user, 'movieId': current_movie_id_2, 'rating': rating_for_movie_2,
             'timestamp': 1111111111}, ignore_index=True)
        recommended_movies = recommendation(8, favourite_movie_name)
        current_movie_id_3 = int(movies.loc[x]['id'])
        ratings_by_users = ratings_by_users.append(
            {'userId': current_user, 'movieId': current_movie_id_3, 'rating': rating_for_movie_3,
             'timestamp': 1111111111}, ignore_index=True)
        recommended_movie_names, recommended_movie_posters = recommendation(current_user, favourite_movie_name)
        st.header("Recommended For You")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.write(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
        with col2:
            st.write(recommended_movie_names[1])
            st.image(recommended_movie_posters[1])
        with col3:
            st.write(recommended_movie_names[2])
            st.image(recommended_movie_posters[2])
        with col4:
            st.write(recommended_movie_names[3])
            st.image(recommended_movie_posters[3])
        with col5:
            st.write(recommended_movie_names[4])
            st.image(recommended_movie_posters[4])
        st.balloons()

