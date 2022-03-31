# pip install streamlit pystan = 2.19.9.1 fbprophet yfinance plotly
import streamlit as st
from datetime import date

from threading import Thread
import threading
import multiprocessing
import string

# Displaying the Words.
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

# Load Packages
import numpy as np
import pandas as pd
from pandas_datareader import data
from datetime import date

import matplotlib.pyplot as plt

import VideoPlaylist

from textblob import TextBlob
from sklearn.manifold import TSNE

import Sentimental_Analysis

# Importing the Necessary lobararies for Google Sheet Integration.
import pandas as pd
import gspread
import df2gspread as d2g

# try:
#     from streamlit.ReportThread import add_report_ctx
#     from streamlit.server.Server import Server
# except Exception:
#     # Streamlit >= 0.65.0
#     from streamlit.report_thread import add_report_ctx
#     from streamlit.server.server import Server

import altair as alt


# from fbprophet import Prophet
# from fbprophet.plot import plot_plotly
# from plotly import graph_objs as go
# from matplotlib.animation import FuncAnimation

from youtubesearchpython import VideosSearch

from streamlit_player import st_player

import os
import logging
from itertools import count

from PIL import Image

import time

import YoutubeAPI

from nltk import word_tokenize
from gensim.models import Word2Vec as w2v
from sklearn.decomposition import PCA


def data_preprocessing(lines, sw = STOPWORDS):
    # remove new lines
    # lines = [line.rstrip('\n') for line in lines]

    # remove punctuations from each line
    lines = [line.translate(str.maketrans('', '', string.punctuation)) for line in lines]


    # Removing the StopWords.    
    res = []
    for line in lines:
        original = line
        line = [w for w in line if w not in sw]
        if len(line) < 1:
            line = original
        res.append(line)
    return res

def tsne_plot(model):
    # Crearting the T-SNE Plot.
    labels = []
    tokens = []

    for word in model.wv.index_to_key:
        tokens.append(model.wv[word])
        labels.append(word)
    # tokens = list(model.wv.key_to_index.vals())
    # tokens
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    fig = plt.figure(figsize=(8, 8)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    st.balloons()
    st.pyplot(fig)
    # plt.show()
    return


def vid():
    global url
    global t
    global Vids
    global vid_search
    global displayed

    return

def change_vid_disp(selected_video):
    selected_video
    st.session_state.selected_video = selected_video
    return

def rerun():
    rerun = True
    return


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# All the required Variables.
url = []
Vids = []
t = {}

def main():
    global url
    global Vids
    global t
    # Starting with the Main App.

    # st.title('Youtube Universe of Comments.')

    welcome_txt = '''Years for now the comment section of Youtube has been plagued with random spam content and the youtube doesn't seem to doing anything about it. We here introduce a new comment section revamped to make the good comments float to the top and the spam ones to linger to the bottom.'''

    _, _, _,  col, _, _, _ = st.columns(7)

    col.title('🙏 ')

    _, col, _ = st.columns(3)

    col.title('**_Welcome_**')

    _, _, _,  col, _, _, _ = st.columns(7)

    col.title('**_to_**')
    st.title('🌎**_Youtube Universe of Comments_**🌏.')

    st.sidebar.title('🌎**_Youtube Universe of Comments_**🌏.')


    with st.sidebar.expander("What is Youtube universe of Comments??", expanded=True):
        st.markdown(welcome_txt)


    # Create a page dropdown
    # page = st.sidebar.selectbox("Choose your page", ["View Video."])

    # st.sidebar.markdown("Clean up the Youtube Section.")
    # st.sidebar.markdown(" ")


    method = st.sidebar.radio("Choose which way you want to get the video :", [
                      "Search the video online.", "Paste the URL of the video."])

    if "rerun" not in st.session_state:
        st.session_state.rerun = False

    if method == "Paste the URL of the video.":
        url = st.text_input(
            "Please enter the url of the video you want to check the comments of.")
    else:
        vid_search = st.text_input(
            "Choose the name of the video you want to search.")
        url = ""
        # seach_vids = YoutubeApI.Search.list(vid_name)
        videosSearch = VideosSearch(vid_search, limit=10)

        # Searching the Video using the URL.

        vids = []
        urls = []

        if vid_search != '':
            for i in range(10):
                Vids.append(videosSearch.result()['result'][i]['title'])
                urls.append(videosSearch.result()['result'][i]['link'])

            for i in range(10):
                t[Vids[i]] = urls[i]

            if "selected_video" not in st.session_state:
                st.session_state.selected_video = Vids[0]

            selected_video = Vids[0]

            selected_video = st.selectbox('Select the Video', Vids)

            # prev = selected_video

            # st.session_state.selected_video

            url = t[selected_video]
            st.subheader(selected_video)


    if url != "":

        # Embed a youtube video
        st_player(url)

        youtube = YoutubeAPI.YoutubeAPI(url)
        # parse video ID from URL.
        youtube.video_id = youtube.get_video_id_by_url(url)
        # make API call to get video info.
        youtube.response = youtube.get_video_details(
            youtube.youtube, id=youtube.video_id)
        # print extracted video infos.
        disp_msg = st.subheader("Extracting the Information about the videos.")
        youtube.get_video_infos(youtube.response)

        thumbnail_url = "http://img.youtube.com/vi/" + \
            youtube.video_id + "/hqdefault.jpg"

        c1, c2, c3 = st.columns(3)

        c2.image(thumbnail_url)

        title = "Title : " + str(youtube.title)
        channel = "Channel : " + str(youtube.channel_title)
        likes = "Likes : " + youtube.like_count
        views = "Views : " + youtube.view_count
        if youtube.comment_count is not None:
            comments = "Comments : " + str(youtube.comment_count)
        else:
            comments = "Comments : (N/A) Disabled "

        day, p_time = youtube.publish_time.split("T")
        pub_time = "Video was Published on " + day + " at " + p_time[:-1]
        hrs = 0
        mins = 0
        if "H" in youtube.duration:
            hrs, min = youtube.duration[2:-1].split("H")
            mins, secs = min.split("M")
        elif "M" in youtube.duration:
            mins, secs = youtube.duration[2:-1].split("M")
        else:
            secs = youtube.duration[2:-1]
        if hrs != 0:
            duration = "Video Duration : " + hrs + " Hours " + \
                mins + " Mins and " + secs + " Secs."
        elif mins != 0:
            duration = "Video Duration : " + mins + " Mins and " + secs + " Secs."
        else:
            duration = "Video Duration : " + secs + " Secs."

        st.subheader(title)
        st.subheader(channel)
        with st.expander("Description :"):
            st.text(youtube.description)

        col1, col2, col3 = st.columns(3)

        col1.subheader(likes)
        col2.subheader(views)
        col3.subheader(comments)

        st.subheader(pub_time)
        st.subheader(duration)

        disp_msg.subheader(
            "Information Extracted Sucessfully. Now onto scrapping the Comments.")

        if youtube.comment_count is not None:
            youtube.scrap_comments(url)
            # disp_msg.subheader("The Comments from the video extracted Sucessfully.")
            disp_msg.text("")

            youtube.df = youtube.df.drop("Comment_id", axis=1)

            st.header("The Top 5 Comments of the Video are :")

            # comments = [for i in range(5) st.expander()]

            for i in range(5):
                name = youtube.df.iloc[i]
                author_name = name["Author Name"]
                comment_body = name["Comment"]

                with st.expander(author_name + " Says : "):
                    st.markdown(comment_body)
                    # blob = TextBlob(comment_body)
                    # lang = blob.detect_language()
                    # st.markdown("lang : ", lang)

            # st.dataframe(youtube.df[["Author Name", "Comment"]])

            text_msg = st.subheader("Analysing the comments.")

            senti_analysis = Sentimental_Analysis.Senti_Analysis()

            text_msg.subheader("Analysis Complete.")

            df = senti_analysis.SentimientAnalysis(youtube.df)

            col1, col2 = st.columns(2)

            col1.text("Top 10 positive Comments : ")

            col2.text("Top 10 Negative Comments : ")

            col1, col2 = st.columns(2)

            col1.dataframe(df.sort_values(by=['Polarity'], ascending=False).head(10)[
                ["Comment", "Polarity", "Subjectivity"]])

            col2.dataframe(df.sort_values(by=['Polarity']).head(10)[
                ["Comment", "Polarity", "Subjectivity"]])

            # Gravity is defined as : G = (Subjectivity)/(2 + Polarity)**2

            df["Gravity"] = (df["Subjectivity"] + 0.1)/(2 + df["Polarity"])**2

            col1, col2 = st.columns(2)

            col1.dataframe(df.sort_values(by=['Gravity']).head(10)[
                ["Comment", "Gravity"]])

            col2.dataframe(df.sort_values(by=['Gravity'], ascending=False).head(10)[
                ["Comment", "Gravity"]])

            # Saving the DataFrame as a CSV file to use later.
            df.to_csv('Data.csv')

            # st.dataframe(df[["Country_Location"]])

            # st.dataframe(df[["Comment", "Polarity", "Subjectivity"]])

            # Displaying the dataframe words.
            # st.dataframe(df)

            # Now Displaying the Given Words into a Wordcloud.
            comment_words = ''
            stopwords = set(STOPWORDS)

            for vals in df.Comment:
                # Typecasting the Value to the string.
                vals = str(vals)

                # splitting the value.
                tokens = vals.split(' ')

                # Converts each token into lowercase
                for i in range(len(tokens)):
                    tokens[i] = tokens[i].lower()
                
                comment_words += " ".join(tokens)+" "

            # Now Generating the WordCloud.
            wordcloud = WordCloud(width = 800, height = 800,
                background_color ='White',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)

            wordcloud.to_file('WordCloud.png')

            # Now Visualizing the WordCloud as an image.
            st.image('WordCloud.png')

            # Now creating the WordEmbedding.
            filtered_lines = data_preprocessing(lines = comment_words, sw = stopwords)

            w = w2v(
            filtered_lines,
            min_count=3,  
            sg = 1,       
            window=7)       


            emb_df = (
                pd.DataFrame(
                    [w.wv.get_vector(str(n)) for n in w.wv.key_to_index],
                    index = w.wv.key_to_index
                )
            )
            # st.dataframe(emb_df.head())

            # tsne_plot(w)

            # Dumping the data into the Spreadsheet.
            # Scopes are the addresses where we want the 
            scope = ['https://spreadsheets.google.com/feeds',
                     'https://www.googleapis.com/auth/drive']
            
            # Now Bringing in the credentials.
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                        'jsonFileFromGoogle.json', scope)

            gc = gspread.authorize(credentials)

            # Spreadsheet Key.
            spreadsheet_key = "1fj3CTi1Px5FuhTtLcrWtm8_-jwT0wZNhJNYdTO4wUog"
            wks_name = 'Master'
            d2g.upload(df, spreadsheet_key, wks_name, credentials=credentials, row_names=True)
            
        else:
            # Then the comments for the video have been disabled.
            st.header(
                "The Comments for this Video have been disabed So unable to proceed.")
            st.header("So, please choose a different video.")


    else:
        st.markdown("Please Enter a Video URL to scrap the comments from.")

    return

main()
