import os
from glob import glob

def printLine(files):
    for f in sorted(files):
        print(f)

path = "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/"
all_files = set(glob(path + "*"))
txt_files = set(glob(path + "*.txt"))
mp4_files = set(glob(path + "*.mp4"))


printLine(all_files)
print("=========================================")
# for path in sorted(all_files - txt_files - mp4_files):
#     if os.path.exists(path + ".mp4"):
#         continue

#     os.system("rm -rf " + path)