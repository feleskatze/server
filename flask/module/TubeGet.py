import os
from pytube import YouTube
import ffmpeg
import datetime

def TubeGet(text):
    YouTubeURL = text.split()[1]
    yt = YouTube(YouTubeURL)
    
    filename = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    video = yt.streams.filter(progressive=True, mime_type='video/mp4').desc().first()
    video.player_config_args['title'] = filename
    video.download('/home/feleskatze/www/flask/tmp/')

    stream = ffmpeg.input('/home/feleskatze/www/flask/tmp/' + filename + '.mp4')
    stream = ffmpeg.output(stream,'/home/feleskatze/www/flask/tmp/' + filename+ '.mp3')
    ffmpeg.run(stream)

    return '/tmp/' + filename+ '.mp3'


if __name__ == '__main__':
    TubeGet('#YtoMP3#\nhttps://www.youtube.com/watch?v=CWSLXXDR8XU')
