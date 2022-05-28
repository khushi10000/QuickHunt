# QuickHunt

Welcome to Quick Hunt 

Commands to run Project is:
pip i -r requriements.txt
streamlit run home.py


It is a Movie Recommendation System which consists of using Hybrid recommendation, Content Recommendation and Popularity Based Recommendation.

PinPoint Recommendation

It collects the user's favorite movie and finds all the movies similar to it by Content Based Recommendation.

In order to eliminate login details of users it is creating a new entry in the dataset of the current user and gets the rating of movies previously watched by the user.

After getting the user entry it analyzes the ratings which the user will give to the movie selected by content based recommendation through SVD.

It sorts the highest rated movies.

Wohoo! PinPoint Recommendation is ready!!

Quick Recommendation

If the user does not want to interact much with the system it just has to select the favorite movie and he/she will get recommendation of the most similar movies by vectorising and applying cosine similarity.

Top Movies 


If the person wants to follow global trends and explore he/she can always come up to Top Movies Recommendation.

It takes into account Minimum Votes, Mean Votes, total number of voters to calculate its popularity and give the top 10 movies according to the dataset.


It’s all about my project. Thanks for your patience.
