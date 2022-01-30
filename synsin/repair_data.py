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

    f = open(out_f + ".txt", "r")
    url = f.readline()
    f.close()

    yt = YouTube(url)
    output_file = yt.streams.get_highest_resolution().download()
    print("Download: {} Compete => {}".format(url, output_file))

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
f = open(path + "video_loc_20220127.txt", "r")
all_file = f.readlines()
f.close()

def download(partial_file):

    for filename in partial_file:
        filename = filename.split("\n")[0]

        try:
            imageSaveFolder = path.split("\n")[0]
            download_dataset(imageSaveFolder, imageSaveFolder, filename)
            print("Download Video and Convert Image Finish !!! (video ID = {})".format(filename))
        except:
            print("Download Video and Convert Image Error !!! (video ID = {})".format(filename))


download(all_file)

# from multiprocessing import Process

# process_num = 8
# count = (len(all_file) // process_num) + 1

# p = [0 for _ in range(process_num)]
# for i in range(process_num):
#     p[i] = Process(target = download, args=(all_file[count * i: count * (i + 1)],))
#     p[i].start()

# for i in range(process_num):
#     p[i].join()
