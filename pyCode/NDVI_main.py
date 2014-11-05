import numpy as np 
import cv2
from matplotlib import pyplot as plt


imgNIR = cv2.imread('b1.jpg',1) #0 gray #1color
imgRGB = cv2.imread('b2.jpg',1)


# (b1,ir,g2) = cv2.split(imgNIR)

(b2,g2,r) = cv2.split(imgRGB)
# ir = np.zeros(r.shape,np.uint8)

ir = cv2.cvtColor(imgNIR, cv2.COLOR_BGR2GRAY)


print ir.shape
print r.shape

cv2.imshow('RGB', imgRGB)
cv2.imshow('RED', r)
# cv2.waitKey(0)
# cv2.close

cv2.imshow('INFRARED', ir) 
# cv2.waitKey(0)

ndvi = np.zeros(ir.shape,np.uint8)

for x in range (0,ir.shape[0]):
	for y in range (0,ir.shape[1]):
		# ndvi[x,y] =  ir[x,y] - r[x,y]
		numerator   = float(ir[x,y]) / 255.0 - float(r[x,y]) / 255.0
		denominator = float(ir[x,y]) / 255.0 + float(r[x,y]) / 255.0
		if (denominator == 0.0):
			fraction = 0.0
			print "fuck guido division by zero exception"
		else:
			fraction  = (numerator / denominator) / 2.0 + 0.5
		ndvi[x,y] =  int(fraction * 255.0) 

		# ndvi[x,y] =  int(float(ir[x,y] - r[x,y]) / float(ir[x,y] + r[x,y]) * 255.0 )

orb = cv2.ORB()

cv2.imshow('NDVI', ndvi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# keypoints1, des1 = orb.detectAndCompute(ir,None)
# keypoints2, des2 = orb.detectAndCompute(g,None)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# matches = bf.match(des1,des2)

# matches = sorted(matches, key = lambda x:x.distance)


# offsetSumx = 0
# offsetSumy = 0

# for m in matches:
# 	offx = keypoints1[matches[0].queryIdx].pt[0] - keypoints2[matches[0].trainIdx].pt[0]
# 	offy = keypoints1[matches[1].queryIdx].pt[1] - keypoints2[matches[1].trainIdx].pt[1]
# 	offsetSumx += offx
# 	offsetSumy += offy

# offset = (offsetSumx/len(matches) , offsetSumy/len(matches))

# print len(matches)

# print offset

# print keypoints1[matches[0].queryIdx].pt


# img3 = cv2.drawMatches(ir,keypoints1,g,keypoints2,matches[:300], None, matchColor = (0,255,0), singlePointColor = (255,0,0), flags=2)
# # img3 = cv2.DRAW_MATCHES_FLAGS_DEFAULT(img1, keypoints1, img1, keypoints2, matches)

# plt.imshow(img3),plt.show()