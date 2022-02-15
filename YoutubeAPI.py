# Importing the Required Packages.
import numpy as np
import pandas as pd
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import urllib.parse as p
import re
import os
import pickle


class YoutubeAPI:
    def __init__(self, url):
        self.youtubeApiKey = "AIzaSyA9TQNt6htXodGXs_lX9mdWVEdmxHOh1do"
        self.youtube = build('youtube', 'v3', developerKey = self.youtubeApiKey)
        # # parse video ID from URL
        # self.video_id = self.get_video_id_by_url(url)
        # # make API call to get video info
        # self.response = self.get_video_details(self.youtube, id=self.video_id)
        # # print extracted video infos
        # self.get_video_infos(self.response)
        # self.scrap_comments(url)
        # parse video ID from URL
        self.video_id = 0
        # make API call to get video info
        self.response = 0
        return 
    
    def get_video_id_by_url(self, url):
        """
        Return the Video ID from the video `url`
        """
        # split URL parts
        parsed_url = p.urlparse(url)
        # get the video ID by parsing the query of the URL
        video_id = p.parse_qs(parsed_url.query).get("v")
        if video_id:
            return video_id[0]
        else:
            raise Exception(f"Wasn't able to parse video URL: {url}")
        
    def get_video_details(self, youtube, **kwargs):
        return youtube.videos().list(
            part="snippet,contentDetails,statistics",
            **kwargs
        ).execute()
    
    def get_video_infos(self, video_response):
        items = video_response.get("items")[0]
        # get the snippet, statistics & content details from the video response
        self.snippet         = items["snippet"]
        self.statistics      = items["statistics"]
        self.content_details = items["contentDetails"]
        # get infos from the snippet
        self.channel_title = self.snippet["channelTitle"]
        self.title         = self.snippet["title"]
        self.description   = self.snippet["description"]
        self.publish_time  = self.snippet["publishedAt"]
        # get stats infos
        self.comment_count = self.statistics["commentCount"]
        self.like_count    = self.statistics["likeCount"]
        # dislike_count = statistics["dislikeCount"]
        self.view_count    = self.statistics["viewCount"]
        # get duration from content details
        self.duration = self.content_details["duration"]
        # duration in the form of something like 'PT5H50M15S'
        # parsing it to be something like '5:50:15'
        self.parsed_duration = re.search(f"PT(\d+H)?(\d+M)?(\d+S)", self.duration).groups()
        self.duration_str = ""
        for d in self.parsed_duration:
            if d:
                self.duration_str += f"{d[:-1]}:"
        self.duration_str = self.duration_str.strip(":")
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
        return 

    def get_comments(self, youtube, **kwargs):
        return youtube.commentThreads().list(
            part="snippet",
            **kwargs
        ).execute()
    
    def scrap_comments(self, url):
        if "watch" in url:
            # that's a video
            video_id = self.get_video_id_by_url(url)
            params = {
                'videoId': video_id, 
                # 'maxResults': 1,
                'order': 'relevance', # default is 'time' (newest)
            }
        else:
            # should be a channel
            channel_id = self.get_channel_id_by_url(url)
            params = {
                'allThreadsRelatedToChannelId': channel_id, 
                # 'maxResults': 2,
                'order': 'relevance', # default is 'time' (newest)
            }
        
        done = False
        comments = 0
        self.df = []
        page_no = 0
        while not done:
            page_no += 1
            # make API call to get all comments from the channel (including posts & videos)
            response = self.get_comments(self.youtube, **params)
            items = response.get("items")
            # print(items)
            # if items is empty, breakout of the loop
            if not items:
                # done = True
                break
            for item in items:
                comments += 1
                # comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comment = item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
                updated_at = item["snippet"]["topLevelComment"]["snippet"]["updatedAt"]
                like_count = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]
                author = item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
                comment_id = item["snippet"]["topLevelComment"]["id"]
                row = [comment_id, author, comment, updated_at, like_count]
                self.df.append(row)
            if "nextPageToken" in response:
                # if there is a next page
                # add next page token to the params we pass to the function
                params["pageToken"] =  response["nextPageToken"]
            else:
                # must be end of comments!!!!
                done = True
                break
        
        self.df = pd.DataFrame(self.df, columns = ['Comment_id','Author Name' ,'Comment', 'Updated_at', 'Like_Count' ])
        
        return page_no, comments