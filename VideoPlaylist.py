import googleapiclient.discovery
from urllib.parse import parse_qs, urlparse

def get_links(url):
    #extract playlist id from url
    links = []
    
    query = parse_qs(urlparse(url).query, keep_blank_values=True)
    playlist_id = query["list"][0]

    print(f'get all playlist items links from {playlist_id}')
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey = "AIzaSyA9TQNt6htXodGXs_lX9mdWVEdmxHOh1do")

    request = youtube.playlistItems().list(
        part = "snippet",
        playlistId = playlist_id,
        maxResults = 50
    )
    response = request.execute()

    playlist_items = []
    while request is not None:
        response = request.execute()
        playlist_items += response["items"]
        request = youtube.playlistItems().list_next(request, response)

    # print(f"total: {len(playlist_items)}")
    # print([ 
    #     f'https://www.youtube.com/watch?v={t["snippet"]["resourceId"]["videoId"]}&list={playlist_id}&t=0s'
    #     for t in playlist_items
    # ])

    liks = list(f'https://www.youtube.com/watch?v={t["snippet"]["resourceId"]["videoId"]}&list={playlist_id}&t=0s'
        for t in playlist_items)

    return liks


# url = 'https://www.youtube.com/playlist?list=PL3D7BFF1DDBDAAFE5'
# get_links(url)