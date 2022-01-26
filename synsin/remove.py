import os
from glob import glob

path = "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/"
paths = glob(path + "*.mp4")

removeNum = 3000
for i in range(removeNum):
    mp4_file = paths[i]
    mp4_folder = paths[i].split(".mp4")[0]
    print(mp4_file)
    print(mp4_folder)

    # os.system("rm {}".format(mp4_file))
    # os.system("rm -rf {}".format(mp4_folder))
