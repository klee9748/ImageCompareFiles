#https://docs.google.com/spreadsheets/d/1sc7P79x82kak7aVGwl2TMKdI_Qex4spJD6ujXdSyXKg/edit#gid=0
#https://docs.google.com/document/d/1Wl8o1j5MmhqRMR6NsbqAKiDN2nVuUWwXkbVO9EEPy7Y/edit

import cv2
import glob
import os
import numpy as np


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        

vidcap = cv2.VideoCapture('Old_Vid.Mov')
success,image = vidcap.read()

count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

old_data = []
files = glob.glob("*jpg")
for myFile in files:
	image = cv2.imread(myFile)
	old_data.append(image)

print('old_data shape:', np.array(X_data).shape)

