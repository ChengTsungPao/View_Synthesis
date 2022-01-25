import os
from glob import glob

def printLine(files):
    for f in files:
        print(f)

path = "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/"
all_files = glob(path + "*")
txt_files = glob(path + "*.txt")
mp4_files = glob(path + "*.mp4")

# printLine(all_files)
printLine(set(all_files) - set(txt_files))
print("===============================================")
printLine(mp4_files)