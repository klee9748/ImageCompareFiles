import os
import numpy as np
import cv2
from glob import glob

def save_frame(video_path, save_dir, gap=10):
    name = video_path.split("/")[-1].split(".")[0]
    save_path = os.path.join(save_dir, name)
    print(name)
    #create_dir(save_path)


if __name__ == "__main__":
    video_paths = glob("videos/*")
    save_dir = "save"

    for path in video_paths:
        save_frame(path, save_dir, gap=10)

print("done")
