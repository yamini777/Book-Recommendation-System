# app.py

import streamlit as st
import pandas as pd
from recommender import hybrid_recommender  # Import the hybrid_recommender function

# Load the dataset with caching to improve performance
@st.cache_data
def load_data():
    try:
        books = pd.read_csv('books.csv')
        ratings = pd.read_csv('ratings.csv')
        book_tags = pd.read_csv('book_tags.csv')
        tags = pd.read_csv('tags.csv')
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        return books, ratings, book_tags, tags, train, test
    except FileNotFoundError as e:
        st.sidebar.error(f"File not found: {e.filename}")
        return None, None, None, None, None, None

def main():
    st.title("📚 Book Recommender System")

    # Sidebar: Informative Write Statements
    st.sidebar.header("About")
    st.sidebar.write("""
        **Welcome to the Book Recommender System!**

        This application provides personalized book recommendations based on your preferences. 
        To get started, enter your User ID and specify your favorite genre, author, or a recent book you've read.
    """)

    # Load data
    books, ratings, book_tags, tags, train, test = load_data()

    # Check if any DataFrame failed to load
    if any(df is None for df in [books, ratings, book_tags, tags, train, test]):
        st.stop()  # Stop the app if data is not loaded properly

    #  Display loaded data sizes for debugging
    st.write("### Data Overview")
    st.write(f"**Books loaded:** {len(books)}")
    st.write(f"**Ratings loaded:** {len(ratings)}")
    st.write(f"**Book Tags loaded:** {len(book_tags)}")
    st.write(f"**Tags loaded:** {len(tags)}")
    st.write(f"**Train set size:** {len(train)}")
    st.write(f"**Test set size:** {len(test)}")

    # Main Area: User Preferences
    st.header("🔍 User Preferences")

    # User ID Input
    user_id = st.number_input("Enter your User ID", min_value=1, value=1, step=1)

    # Favorite Genre Input
    favorite_genre = st.text_input("Enter your favorite genre")

    # Favorite Author Input
    favorite_author = st.text_input("Enter your favorite author")

    # Recent Book Input
    recent_book = st.text_input("Enter the most recent book you've read")

    # Get Recommendations Button
    if st.button("Get Recommendations"):
        # Validate User ID
        if user_id not in ratings['user_id'].unique():
            st.error("User ID not found in the ratings data. Please enter a valid User ID.")
        else:
            with st.spinner('Generating recommendations...'):
                recommendations = hybrid_recommender(
                    user_id=user_id,
                    ratings=ratings,
                    books=books,
                    book_tags=book_tags,
                    tags=tags,
                    favorite_genre=favorite_genre if favorite_genre else None,
                    favorite_author=favorite_author if favorite_author else None,
                    recent_book=recent_book if recent_book else None,
                    n_recommendations=10
                )
        
            if recommendations.empty:
                st.warning("No recommendations found based on your preferences.")
            else:
                st.subheader("📖 Here are your book recommendations:")
                st.dataframe(recommendations)

if __name__ == "__main__":
    main()
