import os
from glob import glob

path = "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/"
paths = glob(path + "*.mp4")

removeNum = 3000
for i in range(removeNum):
    folder = paths.split(".mp4")[0]
    print(paths)
    print(folder)

    # os.system("rm {}".format(paths))
    # os.system("rm -rf {}".format(folder))
