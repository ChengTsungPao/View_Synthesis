# from pytube import YouTube
# import os
# from glob import glob

# def printLine(files):
#     for f in sorted(files):
#         print(f)

# path = "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/"
# all_files = set(glob(path + "*"))
# txt_files = set(glob(path + "*.txt"))
# mp4_files = set(glob(path + "*.mp4"))


# printLine(all_files - txt_files)
# print("=========================================")

# for path in sorted(txt_files):
#     if not os.path.exists(path.split(".txt")[0] + ".mp4"):
#         f = open(path, "r")
#         url = f.readline()
#         f.close()

#         try:
#             yt = YouTube(url)
#             p = yt.streams.get_highest_resolution().download()
#             print("Download: {} Compete => {}".format(url, p))
#             break
#         except:
#             print("Can not download {} !!!".format(path))

####################################################################
from pytube import YouTube
import os
import glob
import cv2
import youtube_dl
import numpy as np


def download_dataset(txt_dir, out_dir, videotxtFilename, stride=1, remove_video=False):
    f = os.path.join(txt_dir, videotxtFilename + '.txt')
    # print("video ID = {}".format(videotxtFilename))
    file_name = f.split('\\')[-1].split('.')[0]  #the file name and remark
    out_f = os.path.join(out_dir,file_name)
    video_txt = open(f)
    content = video_txt.readlines()
    url = content[0]   #the url file
    if not os.path.exists(out_f + '.mp4'):
        # try:
        # ydl_opts = {'outtmpl': '%(id)s.%(ext)s'}
        # with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        #     info = ydl.extract_info(url, download=True)
        #     output_file = ydl.prepare_filename(info)

        f = open(out_f + ".txt", "r")
        url = f.readline()
        f.close()

        yt = YouTube(url)
        output_file = yt.streams.get_highest_resolution().download()
        print("Download: {} Compete => {}".format(url, output_file))

        # except:
        #     print("An exception occurred, maybe because of the downloading limits of youtube.")
    else:
        print('The file {} exists. skip....'.format(out_f + '.mp4'))
        return
    #if video is already downloaded, start extracting frames
    os.makedirs(out_f, exist_ok=True)
    if not os.path.exists(output_file): output_file = output_file.replace('.mp4','.mkv')
    os.rename(output_file, os.path.join(out_f, file_name + '.mp4'))
    line = url
    vidcap = cv2.VideoCapture(os.path.join(out_f, file_name + '.mp4'))
    frame_ind = 1
    frame_file = open(out_f + '/pos.txt','w')
    for num in range(1, len(content), stride):
        line = content[num]
        videoTime = line.split(" ")[0]
        print("Convert Time = {} to Image".format(videoTime))
        frame_file.write(line)
        if line == '\n': break
        #line = video_txt.readline()
        ts = line.split(' ')[0][:-3]  #extract the time stamp
        if ts == '': break
        vidcap.set(cv2.CAP_PROP_POS_MSEC,int(ts))      # just cue to 20 sec. position
        success,image = vidcap.read()
        if success:
            # Cheng Fix
            cv2.imwrite(out_f + '/' + str(videoTime) + '.png', image)     # save frame as JPEG file
            frame_ind += stride
    frame_file.close()
    video_txt.close()
    
    if remove_video:
        os.remove(os.path.join(out_f, file_name + '.mp4'))


path = "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/"
f = open(path + "video_loc_all.txt", "r")
all_file = f.readlines()
f.close()

for index in range(len(all_file)):

    # try:
    # imageSaveFolder = path
    download_dataset(imageSaveFolder, imageSaveFolder, all_file[index])
    print("Download Video and Convert Image Finish !!! (video ID = {})".format(all_file[index]))
    # except:
    #     print("Download Video and Convert Image Error !!! (video ID = {})".format(all_file[index]))