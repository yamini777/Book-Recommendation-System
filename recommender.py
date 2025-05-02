import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

def collaborative_recommender(user_id, ratings, books, n_recommendations=10):
    """
    Collaborative filtering recommendations based on user ratings.
    """
    ratings_cleaned = ratings.groupby(['user_id', 'book_id']).agg({'rating': 'mean'}).reset_index()
    
    # Create user-book matrix
    user_book_matrix = ratings_cleaned.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
    
    # Optionally sample users to reduce size
    sample_size = min(10000, user_book_matrix.shape[0])  # Adjust this sample size as needed
    sampled_user_book_matrix = user_book_matrix.sample(sample_size, random_state=42)

    # Use sparse matrix representation
    sparse_matrix = csr_matrix(sampled_user_book_matrix)

    # Perform SVD
    svd = TruncatedSVD(n_components=20, random_state=42)  # Adjust the number of components
    latent_matrix = svd.fit_transform(sparse_matrix)
    
    # Compute user similarity
    user_similarity = cosine_similarity(latent_matrix)

    user_idx = user_id - 1  # Adjusting index if user_id starts at 1
    
    if user_idx < 0 or user_idx >= user_similarity.shape[0]:
        return pd.DataFrame(columns=['original_title', 'authors'])
    
    # Get similar users
    similar_users = list(enumerate(user_similarity[user_idx]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
    top_users = similar_users[1:11]  # Get top 10 similar users
    top_user_ids = [user[0] + 1 for user in top_users]  # Adjust index back to original user_ids
    
    # Fetch ratings from similar users
    similar_user_ratings = ratings_cleaned[ratings_cleaned['user_id'].isin(top_user_ids)]
    user_rated_books = ratings_cleaned[ratings_cleaned['user_id'] == user_id]['book_id'].tolist()
    
    # Recommend books that similar users rated but the target user hasn't rated
    recommended_books = similar_user_ratings[~similar_user_ratings['book_id'].isin(user_rated_books)]
    top_recommended_books = recommended_books['book_id'].value_counts().head(n_recommendations).index.tolist()
    
    return books[books['book_id'].isin(top_recommended_books)][['original_title', 'authors']]

def content_based_recommender(book_title, books, book_tags):
    """
    Recommend books based on content similarities (tags).
    """
    if book_title:
        matching_books = books[books['original_title'].str.contains(book_title, case=False, na=False)]
        if matching_books.empty:
            return pd.DataFrame(columns=['original_title', 'authors'])
        book = matching_books.iloc[0]
        tag_ids = book_tags[book_tags['goodreads_book_id'] == book['book_id']]['tag_id']
        similar_books_ids = book_tags[book_tags['tag_id'].isin(tag_ids)]['goodreads_book_id'].unique()
        return books[books['book_id'].isin(similar_books_ids)][['original_title', 'authors']].head(10)
    return pd.DataFrame(columns=['original_title', 'authors'])

def hybrid_recommender(user_id, ratings, books, book_tags, tags, favorite_genre=None, favorite_author=None, recent_book=None, n_recommendations=10):
    """
    Hybrid recommender system combining content-based and collaborative filtering.
    """
    content_recs = pd.DataFrame()
    
    if recent_book:
        content_recs = content_based_recommender(recent_book, books, book_tags)
    
    if favorite_genre:
        genre_matching_tags = tags[tags['tag_name'].str.contains(favorite_genre, case=False, na=False)]
        if not genre_matching_tags.empty:
            genre_tag_ids = genre_matching_tags['tag_id'].tolist()
            genre_books_ids = book_tags[book_tags['tag_id'].isin(genre_tag_ids)]['goodreads_book_id'].unique()
            genre_books = books[books['book_id'].isin(genre_books_ids)][['original_title', 'authors']]
            content_recs = pd.concat([content_recs, genre_books], ignore_index=True)
    
    if favorite_author:
        author_books = books[books['authors'].str.contains(favorite_author, case=False, na=False)][['original_title', 'authors']]
        content_recs = pd.concat([content_recs, author_books], ignore_index=True)
    
    content_recs = content_recs.drop_duplicates().head(n_recommendations)
    
    if user_id in ratings['user_id'].unique():
        collaborative_recs = collaborative_recommender(user_id, ratings, books, n_recommendations)
        final_recs = pd.concat([content_recs, collaborative_recs]).drop_duplicates().head(n_recommendations)
    else:
        final_recs = content_recs
    
    return final_recs
