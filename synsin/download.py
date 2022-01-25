from pytube import YouTube

yt = YouTube("")
yt.streams.first().download()