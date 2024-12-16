import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

st.title("TuneGroove Recommender System")

# File uploader with unique key 
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"], key="file_1")

# Check if a file is uploaded
if uploaded_file:
    df_songs = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("### Sample of Uploaded Data", df_songs.head(10))
else:
    try:
        df_songs = pd.read_csv("Last.fm_data.csv")  # Load default file
        st.success("Default file 'Last.fm_data.csv' uploaded successfully!")
    except FileNotFoundError:
        st.error("No file uploaded, and 'Last.fm_data.csv' was not found in the local directory.")
        st.stop()
        
df_songs['Datetime'] = pd.to_datetime(df_songs['Date'] + ' ' + df_songs['Time'])
df_songs['Hour'] = df_songs['Datetime'].dt.hour
df_songs['DayOfMonth'] = df_songs['Datetime'].dt.day


# Sidebar Navigation
st.sidebar.title(" Navigation")
page = st.sidebar.radio("Go to", ["Home","About Project and Overview", "Tunegroove User Insights","Top artists And the TGRS"])

user_track_matrix = df_songs.pivot_table(index='Username', columns='Track', aggfunc='size', fill_value=0)


# Home Page
if page == "Home":
    st.title(" TuneGroove Recommender System ")
    st.write(
        """
        This system (Also known as the TGRS) provides personalized music recommendations using 
        **user-based collaborative filtering**. A default file from our user activity in January ("Last_fm.csv") has already been uploaded to 
        demonstrate the use of the system.
        """
    )
    st.image("C:/Users/ENG Mutua Brian/Downloads/music.jpeg", use_container_width=True)
    st.write("Use the sidebar to navigate through the app.")

    # === About Project Page ===

elif page == "About Project and Overview":
    st.title("About This Project")
    st.write(
        """
        This **Music Recommender System** is built using **Python** and has incorporated the use
        **user-based collaborative filtering**. It recommends tracks to users by finding users with
        similar listening histories, calculating predicted interest scores for each track based on those similarities,
        and then ranking the tracks by their predicted interest scores.
        The system uses a combination of a **similarity matrix** and
        **K Nearest Neighbors** to find similar users and recommend tracks they enjoyed.
        
        ### Key Features:
        - Personalized recommendations
        - Upload your own music dataset(Provided that the file contains all columns present in the default file)
        - Data exploration tools
        - Visuals to track the distribution of user activity
        
        """
    )
   
    st.write(" **Project compiled and done by:** Mutua Brian Kitiku")

#  Data Insights Page
elif page == "Tunegroove User Insights":
    st.title(" Tunegroove User Insights")

    # Show basic data info
    if st.toggle("Display User Data"):
        st.write(df_songs)

    st.write("###  Missing Values")
    missing_values = df_songs.isnull().sum()
    st.write(missing_values)
    st.write(""" 
            The 12 missing values present in the Album column may be due to 
             some tracks in the dataset not having any album. This is common as 
             some artists release music as singles rather than albums.
             """ )

    st.write("###  Duplicates in the Data")
    duplicates = df_songs.duplicated().sum()
    st.write(f"Number of Duplicated Rows: {duplicates}")
    st.write(""" 
             The absence of duplicated rows in the dataset shows that no user re-listened to
             any given track at any given time in the dataset.
             """ )

    st.write("###  User Activity")
    user_activity = df_songs['Username'].value_counts()
    st.bar_chart(user_activity.head(10))
    
    st.write(""" 
             Among the 10 users who listened to Last.fm, User Babs_05 was the most active
             listener while User Jajo was the least active user.
             """ )
    
    st.write("###  Top 20 Most Popular Artists")
    top_artists = df_songs['Artist'].value_counts().head(20)
    st.bar_chart(top_artists)

    # User Song Count Distribution
    user_song_count = df_songs.groupby("Username")["Track"].nunique().reset_index()
    user_song_count.columns = ["Username", "unique_songs_played"]

    st.write("### 🎧 Distribution of Unique Songs Played per User")
    st.bar_chart(user_song_count.set_index("Username")["unique_songs_played"])

    st.write("### Listening habits per hour")
    st.bar_chart(df_songs['Hour'].value_counts().sort_index())

    st.write("### Listening habits per day")
    st.bar_chart(df_songs['DayOfMonth'].value_counts().sort_index())


elif page == "Top artists And the TGRS":
    st.title("View Top Artists and the TGRS")
    st.write("Dear user are you still tuned in?, **your monthly wrapped is finally here!!**")
    st.write("")
    st.write("""Hope u wont be disappointed with your **top artists of the month.** And while your at it
    why not give our new **TGRS** a go, you will love it and **you will love the tunes even more!!**
     """)
    st.write("### Input your Username to proceed")

    username = st.text_input("Enter Username to continue")
    Login_button = st.button("Log in")

    if username:
        if username not in df_songs['Username'].unique():
            st.error(f"Username '{username}' not available. Please try again.")
        else:
            # options for user
            view_option = st.radio("What would you like to do?", ["View Top Artists", "TGRS Track Recommendations"])

            if view_option == "View Top Artists":
                # Option to select how many top artists to view
                top_n = st.slider("Select how many top artists to view", min_value=1, max_value=20, value=1)
                
                # Button to display top artists
                show_top_artists_button = st.button("Show Top Artists")
            
                if show_top_artists_button:
                    # Group the data by 'Username' and count the top N most listened-to artists
                    most_listened_df = (
                        df_songs.groupby('Username')['Artist']
                        .apply(lambda x: x.value_counts().nlargest(top_n))
                        .reset_index(name='Times Played')
                    )

                    # Rename the 'level_1' column to 'Artist'
                    most_listened_df.rename(columns={'level_1': 'Artist'}, inplace=True)

                    # Filter to display the current user's top artists
                    user_top_artists = most_listened_df[most_listened_df['Username'] == username]

                    if not user_top_artists.empty:
                        st.write(f"### Top {top_n} Most Listened Artists for {username}")
                        # Display the top artists in a DataFrame with a column called 'Artist'
                        st.dataframe(user_top_artists[['Artist', 'Times Played']])
                    else:
                        st.warning(f"No artist data found for {username}.")
    
            elif view_option == "TGRS Track Recommendations":
                top_rec = st.slider("How many recommendations would you like to add to your current playlist?", min_value=1, max_value=20, value=1)
                submit_button = st.button("Get TGRS Recommendations")

                if submit_button:
                    with st.spinner("Fetching Recommendations....."):
                        user_track_matrix = df_songs.pivot_table(index='Username', columns='Track', aggfunc='size', fill_value=0)

                        # Train the KNN model on the user-track matrix
                        knn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
                        knn_model.fit(user_track_matrix)

                        def recommend_tracks(user, user_track_matrix, knn_model, top_n=5):
                            """ Recommends tracks for a user using K-Nearest Neighbors (KNN) approach. """
                            
                            if user not in user_track_matrix.index:
                                st.error(f"Username '{user}' not found.")
                                return []

                            # Get the target user's track listening history
                            user_data = user_track_matrix.loc[user].values.reshape(1, -1)
                            
                            # Get the indices and distances of the 3 nearest neighbors
                            distances, indices = knn_model.kneighbors(user_data)
                            
                            # Get the usernames of the nearest neighbors
                            similar_users = user_track_matrix.index[indices.flatten()]
                            
                            # Get user's play history (1 = played, 0 = not played)
                            user_track_history = user_track_matrix.loc[user]
                            
                            # Tracks that the user has not listened to
                            track_scores = np.zeros(len(user_track_matrix.columns))

                            for neighbor in similar_users:
                                neighbor_history = user_track_matrix.loc[neighbor]
                                track_scores += neighbor_history.values  # Sum neighbor's play history
                            

                            # Exclude tracks the user has already played
                            for i, track in enumerate(user_track_matrix.columns):
                                if user_track_history[track] > 0:
                                    track_scores[i] = 0  # Do not recommend tracks already played

                            # Rank tracks by highest score
                            recommended_tracks = [track for track, score in sorted(zip(user_track_matrix.columns, track_scores), key=lambda x: -x[1])]
                            
                            return recommended_tracks[:top_n]

                    recommendations = recommend_tracks(username, user_track_matrix, knn_model, top_rec)

                    if recommendations:
                        st.success(f" TGRS Recommendations:")
                        for i, track in enumerate(recommendations, start=1):
                            st.write(f"{i}. {track}")
                    else:
                        st.warning(f"No recommendations found for {username}.")


                        


                    
    





