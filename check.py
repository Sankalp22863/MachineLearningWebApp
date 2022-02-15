import VideoPlaylist

url = 'https://www.youtube.com/playlist?list=PL3D7BFF1DDBDAAFE5'

links = VideoPlaylist.get_links(url)

print(links)
