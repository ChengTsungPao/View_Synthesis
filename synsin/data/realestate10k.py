# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch.utils.data as data

# Cheng Fix
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from utils.geometry import get_deltas


############################################################################################################
# Cheng Fix
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
############################################################################################################


class RealEstate10K(data.Dataset):
    """ Dataset for loading the RealEstate10K. In this case, images are randomly 
    chosen within a video subject to certain constraints: e.g. they should 
    be within a number of frames but the angle and translation should
    vary as much as possible.
    """

    def __init__(
        self, dataset, opts=None, num_views=2, seed=0, vectorize=False
    ):
        # Now go through the dataset
        self.imageset = np.loadtxt(
            opts.train_data_path + "/frames/%s/video_loc.txt" % "train",
            dtype=np.str,
        )

        if dataset == "train":
            self.imageset = self.imageset[0 : int(0.8 * self.imageset.shape[0])]
        else:
            self.imageset = self.imageset[int(0.8 * self.imageset.shape[0]) :]

        self.rng = np.random.RandomState(seed)
        self.base_file = opts.train_data_path

        self.num_views = num_views

        self.input_transform = Compose(
            [
                Resize((opts.W, opts.W)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.offset = np.array(
            [[2, 0, -1], [0, -2, 1], [0, 0, -1]],  # Flip ys to match habitat
            dtype=np.float32,
        )  # Make z negative to match habitat (which assumes a negative z)

        self.dataset = "train"

        self.K = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.invK = np.linalg.inv(self.K)

        self.ANGLE_THRESH = 5
        self.TRANS_THRESH = 0.15

    def __len__(self):
        return 2 ** 31

    def __getitem_simple__(self, index):
        index = self.rng.randint(self.imageset.shape[0])
        # index = index % self.imageset.shape[0]
        # Load text file containing frame information
        frames = np.loadtxt(
            self.base_file
            + "/frames/%s/%s.txt" % (self.dataset, self.imageset[index])
        )

        image_index = self.rng.choice(frames.shape[0], size=(1,))[0]

        rgbs = []
        cameras = []
        for i in range(0, self.num_views):
            t_index = max(
                min(
                    image_index + self.rng.randint(16) - 8, frames.shape[0] - 1
                ),
                0,
            )

            image = Image.open(
                self.base_file
                + "/frames/%s/%s/" % (self.dataset, self.imageset[index])
                + str(int(frames[t_index, 0]))
                + ".png"
            )
            rgbs += [self.input_transform(image)]

            intrinsics = frames[t_index, 1:7]
            extrinsics = frames[t_index, 7:]

            origK = np.array(
                [
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            K = np.matmul(self.offset, origK)

            origP = extrinsics.reshape(3, 4)
            P = np.matmul(K, origP)  # Merge these together to match habitat
            P = np.vstack((P, np.zeros((1, 4), dtype=np.float32))).astype(
                np.float32
            )
            P[3, 3] = 1

            Pinv = np.linalg.inv(P)

            cameras += [
                {
                    "P": P,
                    "OrigP": origP,
                    "Pinv": Pinv,
                    "K": self.K,
                    "Kinv": self.invK,
                }
            ]
        
        return {"images": rgbs, "cameras": cameras}

    def __getitem__(self, index):
        while True:
            index = self.rng.randint(self.imageset.shape[0])
            # index = index % self.imageset.shape[0]
            # Load text file containing frame information

            # Cheng fx
            # print("RealEstate10K Loading ... (__getitem__)")
            frames = np.loadtxt(
                self.base_file
                + "/frames/%s/%s.txt" % (self.dataset, self.imageset[index]),
                skiprows=1
            )

            # # Cheng fx

            # Random Data Input
            import os
            import matplotlib.pyplot as plt
            # for line in frames:
            #     try:
            #         time = line[0]
            #         imageSaveFolder = self.base_file + "/frames/%s/%s/" % (self.dataset, self.imageset[index])
            #         if not os.path.exists(imageSaveFolder):
            #             os.makedirs(imageSaveFolder)
            #         if not os.path.exists(imageSaveFolder + str(int(time)) + ".png"):
            #             plt.imsave(imageSaveFolder + str(int(time)) + ".png", np.random.rand(500, 500, 3))
            #     except:
            #         pass

            try:
                imageSaveFolder = self.base_file + "/frames/%s/" % (self.dataset)
                print("video ID = {}".format(self.imageset[index]))
                download_dataset(imageSaveFolder, imageSaveFolder, self.imageset[index])
                print("Download Video and Convert Image Finish !!!")
                break
            except:
                print("video ID = {}".format(self.imageset[index]))
                print("Download Video and Convert Image Error !!!")
                # pass

        

        image_index = self.rng.choice(frames.shape[0], size=(1,))[0]

        # Chose 15 images within 30 frames of the iniital one
        image_indices = self.rng.randint(80, size=(30,)) - 40 + image_index
        image_indices = np.minimum(
            np.maximum(image_indices, 0), frames.shape[0] - 1
        )

        # Look at the change in angle and choose a hard one
        angles = []
        translations = []
        for viewpoint in range(0, image_indices.shape[0]):
            orig_viewpoint = frames[image_index, 7:].reshape(3, 4)
            new_viewpoint = frames[image_indices[viewpoint], 7:].reshape(3, 4)
            dang, dtrans = get_deltas(orig_viewpoint, new_viewpoint)

            angles += [dang]
            translations += [dtrans]

        angles = np.array(angles)
        translations = np.array(translations)

        mask = image_indices[
            (angles > self.ANGLE_THRESH) | (translations > self.TRANS_THRESH)
        ]

        rgbs = []
        cameras = []
        # Cheng Fix
        # for i in range(0, self.num_views):
        i = 0
        while i < self.num_views:

            if i == 0:
                t_index = image_index
            elif mask.shape[0] > 5:
                # Choose a harder angle change
                t_index = mask[self.rng.randint(mask.shape[0])]
            else:
                t_index = image_indices[
                    self.rng.randint(image_indices.shape[0])
                ]
            # Cheng Fix
            try:
                image = Image.open(
                    self.base_file
                    + "/frames/%s/%s/" % (self.dataset, self.imageset[index])
                    + str(int(frames[t_index, 0]))
                    + ".png"
                ).convert('RGB')
                rgbs += [self.input_transform(image)]

            except:
                continue

            intrinsics = frames[t_index, 1:7]
            extrinsics = frames[t_index, 7:]

            origK = np.array(
                [
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            K = np.matmul(self.offset, origK)

            origP = extrinsics.reshape(3, 4)
            P = np.matmul(K, origP)  # Merge these together to match habitat
            P = np.vstack((P, np.zeros((1, 4), dtype=np.float32))).astype(
                np.float32
            )
            P[3, 3] = 1

            Pinv = np.linalg.inv(P)

            cameras += [
                {
                    "P": P,
                    "Pinv": Pinv,
                    "OrigP": origP,
                    "K": self.K,
                    "Kinv": self.invK,
                }
            ]

            i += 1

        return {"images": rgbs, "cameras": cameras}

    def totrain(self, epoch):
        self.imageset = np.loadtxt(
            self.base_file + "/frames/%s/video_loc.txt" % "train", dtype=np.str
        )
        self.imageset = self.imageset[0 : int(0.8 * self.imageset.shape[0])]
        self.rng = np.random.RandomState(epoch)

    def toval(self, epoch):
        self.imageset = np.loadtxt(
            self.base_file + "/frames/%s/video_loc.txt" % "train", dtype=np.str
        )
        self.imageset = self.imageset[int(0.8 * self.imageset.shape[0]) :]
        self.rng = np.random.RandomState(epoch)


class RealEstate10KConsecutive(data.Dataset):
    """ Dataset for loading the RealEstate10K. In this case, images are 
    consecutive within a video, as opposed to randomly chosen.
    """

    def __init__(
        self, dataset, opts=None, num_views=2, seed=0, vectorize=False
    ):
        # Now go through the dataset

        self.imageset = np.loadtxt(
            opts.train_data_path + "/frames/%s/video_loc.txt" % "train",
            dtype=np.str,
        )

        if dataset == "train":
            self.imageset = self.imageset[0 : int(0.8 * self.imageset.shape[0])]
        else:
            self.imageset = self.imageset[int(0.8 * self.imageset.shape[0]) :]

        self.rng = np.random.RandomState(seed)
        self.base_file = opts.train_data_path

        self.num_views = num_views

        self.input_transform = Compose(
            [
                Resize((opts.W, opts.W)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.offset = np.array(
            [[2, 0, -1], [0, -2, 1], [0, 0, -1]],  # Flip ys to match habitat
            dtype=np.float32,
        )  # Make z negative to match habitat (which assumes a negative z)

        self.dataset = "train"

        self.K = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.invK = np.linalg.inv(self.K)

        self.ANGLE_THRESH = 5
        self.TRANS_THRESH = 0.15

    def __len__(self):
        return 2 ** 31

    def __getitem__(self, index):
        index = self.rng.randint(self.imageset.shape[0])
        # Load text file containing frame information
        
        # Cheng fx
        frames = np.loadtxt(
            self.base_file
            + "/frames/%s/%s.txt" % (self.dataset, self.imageset[index]),
            skiprows=1
        )

        image_index = self.rng.choice(
            max(1, frames.shape[0] - self.num_views), size=(1,)
        )[0]

        image_indices = np.linspace(
            image_index, image_index + self.num_views - 1, self.num_views
        ).astype(np.int32)
        image_indices = np.minimum(
            np.maximum(image_indices, 0), frames.shape[0] - 1
        )

        rgbs = []
        cameras = []
        for i in range(0, self.num_views):
            t_index = image_indices[i]
            image = Image.open(
                self.base_file
                + "/frames/%s/%s/" % (self.dataset, self.imageset[index])
                + str(int(frames[t_index, 0]))
                + ".png"
            )
            rgbs += [self.input_transform(image)]

            intrinsics = frames[t_index, 1:7]
            extrinsics = frames[t_index, 7:]

            origK = np.array(
                [
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            K = np.matmul(self.offset, origK)

            origP = extrinsics.reshape(3, 4)
            np.set_printoptions(precision=3, suppress=True)
            P = np.matmul(K, origP)  # Merge these together to match habitat
            P = np.vstack((P, np.zeros((1, 4), dtype=np.float32))).astype(
                np.float32
            )
            P[3, 3] = 1

            Pinv = np.linalg.inv(P)

            cameras += [
                {
                    "P": P,
                    "Pinv": Pinv,
                    "OrigP": origP,
                    "K": self.K,
                    "Kinv": self.invK,
                }
            ]

        return {"images": rgbs, "cameras": cameras}
