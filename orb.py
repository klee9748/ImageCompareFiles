import numpy as np
import cv2
import matplotlib.pyplot as plt

class Orb:
	#initializer = teach python what orb is 
	def __init__ (self, offset_indexes, X_data, Y_data):
		#[4, 5, 0, 1, 2, 3]
		self.offset_indexes = offset_indexes
		self.X_data = X_data
		self.Y_data = Y_data

	def	create_orb(self):
		for i in range(0, len(self.X_data)):
			image_orb = cv2.ORB_create()
			kp1, des1 = image_orb.detectAndCompute(self.X_data[i], None)
			kp2, des2 = image_orb.detectAndCompute(self.Y_data[self.offset_indexes[i]], None)

			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			matches = bf.match(des1, des2)
			matches = sorted(matches, key = lambda x:x.distance)

			number_keypoints = 0
			if len(kp1) <= len(kp2):
				number_keypoints = len(kp1)
			else:
				number_keypoints = len(kp2)

			print("Keypoints 1ST Image: " + str(len(kp1)))
			print("Keypoints 2ND Image: " + str(len(kp2)))
			print("GOOD Matches:", len(self.offset_indexes))
			print("How good it's the match: ", len(self.offset_indexes) / number_keypoints * 100)

			good_match = len(self.offset_indexes) / number_keypoints * 100
			
			img3 = cv2.drawMatches(self.X_data[i],kp1,self.Y_data[self.offset_indexes[i]],kp2,matches[:10],None, flags=2)
			plt.imshow(img3)
			plt.show()

		#Now Calcuate the percent difference


#class surf:
#	print("Print Surf")
# class sift:
# 	sift = cv2.SIFT_create()
#     kp_1, desc_1 = sift.detectAndCompute(X_data[x], None)
#     kp_2, desc_2 = sift.detectAndCompute(Y_data[y], None)

#     index_params = dict(algorithm=0, trees=5)
#     search_params = dict()
#     flann = cv2.FlannBasedMatcher(index_params, search_params)

#     matches = flann.knnMatch(desc_1, desc_2, k=2)

#     good_points = []

#     for m, n in matches:
#       if m.distance < 0.6*n.distance:
#         good_points.append(m)

#     # Define how similar they are
#     number_keypoints = 0
#     if len(kp_1) <= len(kp_2):
#       number_keypoints = len(kp_1)
#     else:
#       number_keypoints = len(kp_2)


#     print("Keypoints 1ST Image: " + str(len(kp_1)))
#     print("Keypoints 2ND Image: " + str(len(kp_2)))
#     print("GOOD Matches:", len(good_points))
#     print("How good it's the match: ", len(good_points) / number_keypoints * 100)

#     good_match = len(good_points) / number_keypoints * 100
#     comparison_data[x][y] = good_match
