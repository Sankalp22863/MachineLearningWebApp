from youtubesearchpython import VideosSearch

videosSearch = VideosSearch('Carry Minati', limit = 1)

print(videosSearch.result()['result'][0][])