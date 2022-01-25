from pytube import YouTube

f = open("/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/0003a9bce989e532.txt", "r")
url = f.readline()
f.close()


yt = YouTube(url)
yt.streams.first().download()

print("Download: {} Compete".format(url))