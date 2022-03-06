# pip install streamlit pystan = 2.19.9.1 fbprophet yfinance plotly
import streamlit as st
from datetime import date

from threading import Thread
import threading
import multiprocessing

# Load Packages
import numpy as np
import pandas as pd
from pandas_datareader import data
from datetime import date

import matplotlib.pyplot as plt

import VideoPlaylist

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

# from youtubesearchpython import VideosSearch

from streamlit_player import st_player

import os
import logging
from itertools import count

from PIL import Image

import time

import YoutubeAPI



def vid():
	global url 
	global t
	global Vids
	global vid_search
	global displayed
	
	if vid_search != '':
		for i in range(10):
			Vids.append(videosSearch.result()['result'][i]['title'])
			url.append(videosSearch.result()['result'][i]['link'])
		
		for i in range(10):
			t[Vids[i]] = url[i]

		selected_video = st.selectbox('Select the Video', Vids)

		prev = selected_video

		url = t[selected_video]
		st.subheader(selected_video)

		# Embed a youtube video
		st_player(url)
		
		displayed = True

	return 





# All the required Variables.

url = []
Vids = []
t = {}





# Starting with the Main App.

st.title('Youtube Comment Analyzer.')

# Create a page dropdown 
# page = st.sidebar.selectbox("Choose your page", ["View Video."]) 

# st.sidebar.markdown("Clean up the Youtube Section.")
# st.sidebar.markdown(" ")

# Displaying the df.

# vid_search = st.text_input("Search a Youtube Video")
# vid_search = st.text_input("Copy and paste the link of the URL of the youtube Video.")
# videosSearch = VideosSearch(vid_search, limit = 10)

st.selectbox("Choose which way you want to get the video :", ["Search the video online.", "Paste the URL of the video."])

url = st.text_input("Please enter the url of the video you want to check the comments of.")


if url != "":	
	youtube = YoutubeAPI.YoutubeAPI(url)
	# parse video ID from URL.
	youtube.video_id = youtube.get_video_id_by_url(url)
	# make API call to get video info.
	youtube.response = youtube.get_video_details(youtube.youtube, id=youtube.video_id)
	# print extracted video infos.
	disp_msg = st.subheader("Extracting the Information about the videos.")
	youtube.get_video_infos(youtube.response)

	# Displaying the Video information.
	# print(f"""\
        # Title: {title}
        # Description: {description}
        # Channel Title: {channel_title}
        # Publish time: {publish_time}
        # Duration: {duration_str}
        # Number of comments: {comment_count}
        # Number of likes: {like_count}
        # Number of views: {view_count}
        # """)

	thumbnail_url = "http://img.youtube.com/vi/" + youtube.video_id +"/hqdefault.jpg"

	c1, c2, c3 = st.columns(3)

	c2.image(thumbnail_url)
	
	title = "Title : " + str(youtube.title)
	channel = "Channel : " + str(youtube.channel_title)
	likes = "Likes : " + youtube.like_count
	views = "Views : " + youtube.view_count
	comments = "Comments : " + youtube.comment_count

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
		duration = "Video Duration : " + hrs + " Hours " + mins + " Mins and " + secs + " Secs."
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
	

	disp_msg.subheader("Information Extracted Sucessfully. Now onto scrapping the Comments.")
	youtube.scrap_comments(url)
	# disp_msg.subheader("The Comments from the video extracted Sucessfully.")
	disp_msg.text("")

	youtube.df = youtube.df.drop("Comment_id", axis = 1)

	for i in range(5):
		name = youtube.df.iloc[i]
		author_name = name["Author Name"]
		comment_body = name["Comment"]
# 		c1, c2 = columns(2)
		st.subheader(author_name)
		st.text(comment_body)

	# st.dataframe(youtube.df[["Author Name", "Comment"]])
	
else:
	st.markdown("Please Enter a Video URL to scrap the comments from.")

