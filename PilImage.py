import cv2
import glob
import numpy as np
from PIL import Image, ImageChops
X_data = []
Y_data = []
files = glob.glob ("/Users/klee22/Desktop/save/New_Vid/*.png")
files2 = glob.glob ("/users/klee22/Desktop/save/Old_vid/*.png")


numpic_x = 97
numpic_y = 97
comparison_data = np.zeros(shape=(97, 97)) #create an empty grid that has space for the data from the for loop

x = 0
for orignial_image in X_data:
  orignial_pics = Image.open(files2)
  X_data.append(orignial_pics)
  
  reference_pics = Image.open(files)
  Y_data.append(reference_pics)

  comparison_data = np.zeros(shape=(numpic_x, numpic_y))
  y = 0
  for new_image in Y_data:
      diff = ImageChops.difference(files[0], files2[0])
      print(diff.getbox())
        #comparison_score = cv2.SIFT_create(ref_image, new_image)
        
        #comparison_data[x][y] = comparison_score #comparison_data[x][y] -> location
      y += 1
  x += 1
print(X_data,Y_data)


