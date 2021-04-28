import cv2
import glob
import numpy as np
#from PIL import Image, ImageChops
#from random import randrange
#import matplotlib.pyplot as plt

X_data = []
Y_data = []
files = glob.glob ("/Users/klee22/Desktop/save/New_Vid/*.png")
files2 = glob.glob ("/users/klee22/Desktop/save/old_vid/*.png")

#Reads the Files as cv2
for myFile in files:    
  image = cv2.imread (myFile, cv2.IMREAD_COLOR) #enumerated constants
  X_data.append (image)

for myFile2 in files2:
  image2 = cv2.imread (myFile2, cv2.IMREAD_COLOR)
  Y_data.append (image2)

# create multi-dim array by providing shape
print('X_data shape:', np.array(X_data).shape) 
print('Y_data shape:', np.array(Y_data).shape) 


numpic_x = len(X_data)
numpic_y = len(Y_data)
comparison_data = np.zeros(shape=(numpic_x, numpic_y)) #create an empty grid that has space for the data from the for loop
print(X_data[3])

print(comparison_data)
x = 0
for ref_image in X_data:
  y = 0
  for new_image in Y_data:
    if X_data[x].shape == Y_data[y].shape:
      print("The images have same size and channels")
      difference = cv2.subtract(X_data[x], Y_data[y])
      b, g, r = cv2.split(difference)
      if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("The images are completely Equal")
      else:
          print("The images are NOT equal")
  y += 1
x += 1

