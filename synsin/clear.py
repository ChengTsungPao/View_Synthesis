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
        print(path)
        count += 1

    if count == 5:
        break

# for path in sorted(all_files - txt_files - mp4_files):
#     if not os.path.exists(path + ".mp4"):
#         continue

#     os.system("scp -r {} abaozheng6@140.109.21.232:/home/abaozheng6/Temp".format(path))