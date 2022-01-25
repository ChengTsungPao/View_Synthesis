from pytube import YouTube
import os
from glob import glob

def printLine(files):
    for f in sorted(files):
        print(f)

path = "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/"
all_files = set(glob(path + "*"))
txt_files = set(glob(path + "*.txt"))
mp4_files = set(glob(path + "*.mp4"))


printLine(all_files - txt_files)
print("=========================================")

count = 0
for path in sorted(txt_files):
    if not os.path.exists(path.split(".txt")[0] + ".mp4"):
        f = open(path, "r")
        url = f.readline()
        f.close()

        try:
            yt = YouTube(url)
            yt.streams.first().download()
            print("Download: {} Compete".format(url))
            break
        except:
            print("Can not download {} !!!".format(path))