import os
from glob import glob

def printLine(files):
    for f in files:
        print(f)

path = "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/"
all_files = glob(path + "*")
mp4_files = glob(path + "*.mp4")

print(all_files)
print(mp4_files)