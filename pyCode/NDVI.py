import numpy as np
import cv2
import os
import datetime
from matplotlib import pyplot as plt
from win32api import GetSystemMetrics



# Mouse Listener for the left image
def LeftIMGclick(event,x,y,flags,param):
	global RGB
	global POINTS
	if event == cv2.EVENT_LBUTTONDOWN:
		if RGB == False:
			print "not RGB now"
			return
		else:
			RGBpoints.append((x,y))
			POINTS+=1
			RGB = False
			cv2.circle(left, (x,y), 7, (255,255,0))
			cv2.imshow("RGB",  left)

# Mouse Listener for the right image
def RightIMGclick(event,x,y,flags,param):
	global RGB
	global POINTS
	if event == cv2.EVENT_LBUTTONDOWN:
		if RGB == True:
			print "not IR now"
			return
		else:
			IRpoints.append((x,y))
			POINTS+=1
			RGB = True
			cv2.circle(right, (x,y), 7, (255,255,0))
			cv2.imshow("IR",  right)
			

# Main function for setting the manual warping features
def collectPairs():

	global left
	global right

	cv2.namedWindow("RGB")
	cv2.namedWindow("IR")

	# register callbacks
	cv2.setMouseCallback("RGB", LeftIMGclick)
	cv2.setMouseCallback("IR", RightIMGclick)

	#show 2 images
	cv2.imshow("RGB", left)
	cv2.imshow("IR",  right)
	while (True):
		if POINTS == 8:
			global IRpoints
			global RGBpoints
			IRpoints = np.array(IRpoints)
			RGBpoints = np.array(RGBpoints)


			H, mask = cv2.findHomography( IRpoints, RGBpoints, cv2.RANSAC );

			H = np.array(H)

			return H

		if cv2.waitKey(0) & 0xFF == ord("q"):
			cv2.destroyAllWindows()
			return None    

# This warpes the infrared image to the red image with the homography H
def warpAndCropImages(redImage, infraredImage, H):

	WIDTH = redImage.shape[1] 
	HEIGHT = redImage.shape[0] 
	IMGSIZE = (WIDTH, HEIGHT)

	# Warp the IR image on the RED image
	warpedIR = np.zeros(redImage.shape)
	warpedIR = cv2.warpPerspective(infraredImage, H, IMGSIZE, warpedIR, cv2.INTER_LINEAR)

	# Calculate biggest rectangle
	irCorners = np.zeros([4,2], dtype="float32")

	irCorners[0] = [0.0  ,0.0   ]        # 0 top Left
	irCorners[1] = [0.0  ,HEIGHT]        # 1 bot Left
	irCorners[2] = [WIDTH,0.0   ]        # 2 top Right
	irCorners[3] = [WIDTH,HEIGHT]        # 3 bot Right

	# Transform IR Corners in Projected Space
	irCorners = np.array([irCorners])
	irWarpedCorers = cv2.perspectiveTransform(irCorners, H)

	# Get Biggest Bounding Box
	boundLeft  = max(0     ,irWarpedCorers[0][0][0], irWarpedCorers[0][1][0])
	boundRight = min(WIDTH ,irWarpedCorers[0][2][0], irWarpedCorers[0][3][0])
	boundTop   = max(0     ,irWarpedCorers[0][0][1], irWarpedCorers[0][2][1])
	boundBot   = min(HEIGHT,irWarpedCorers[0][1][1], irWarpedCorers[0][3][1])

	# Corner Points
	topLeft = (boundLeft, boundTop)
	botRight = (boundRight, boundBot)

	ROI = (topLeft, botRight)

	# Crop the input Images
	croppedIR = warpedIR[boundTop:boundBot, boundLeft:boundRight]
	croppedR = redImage[boundTop:boundBot, boundLeft:boundRight]

	return croppedR, croppedIR, True


# Transforms the NDVI into the Blue to Green grading
def colorGradeBG(ndvi):

	# red channel
    red = np.zeros(ndvi.shape,np.float32)

    # blue channel
    blue = cv2.divide(ndvi, 2)
    blue = cv2.subtract(0.5, blue)

    # green channel
    green = cv2.add(ndvi, 1.0)
    green = cv2.divide(green, 2.0)

    gradedNDVI = cv2.merge((blue,green,red))

 
    return gradedNDVI

def colorGradeBGR(ndvi):
    
    maskGreater = cv2.compare(ndvi, 0.0, cv2.CMP_GE)

    maskLess = cv2.compare(ndvi, 0.0, cv2.CMP_LT)


    ## greater than 0 part
    # red
    red = np.zeros(ndvi.shape,np.float32)#
    
    # blue is 0
    blue = cv2.add(red,0.0, mask = maskGreater) #, dtype=np.float32)
    
    # red
    red = cv2.add(ndvi,0.0, mask = maskGreater)
    red = cv2.multiply(ndvi,4.0)

    # green
    green = cv2.subtract(2.0, red, mask = maskGreater)

    ## less than 0 part

    # red is 0

  	# blue
    blue = cv2.add(blue, 1.0, mask = maskLess)
    
    # green
    lowgreen = cv2.add(ndvi,1.0, mask = maskLess)
    green = cv2.add(green, lowgreen)
   
    # clamp to [0,1]
    red = cv2.min(red,1)
    red = cv2.max(red,0)  
    green = cv2.min(green,1)
    green = cv2.max(green,0)   
    blue = cv2.min(blue,1)
    blue = cv2.max(blue,0) 
    gradedNDVI = cv2.merge((blue,green,red))

    return gradedNDVI


# calculates the NDVI from 2 red images
# Frames have to be aligned
def calculateNDVI(REDframe, IRframe, grading):

	# convert to floating point precision and scale down to [0;1]
    fRED = REDframe.astype(np.float32)
    fRED = cv2.divide(fRED, 255.0)
    
    fIR = IRframe.astype(np.float32)
    fIR = cv2.divide(fIR, 255.0)

    # calc the ndvi equation
    numerator = cv2.subtract(fIR, fRED)
    denumerator = cv2.add(fIR, fRED)
    fraction = cv2.divide(numerator, denumerator)
    
    NDVIframe = fraction


    # apply additional grading
    if grading == "BG":
    	print "BG"
        NDVIframeG = colorGradeBG(NDVIframe)
    elif grading == "BGR":
    	print "BGR"
        NDVIframeG = colorGradeBGR(NDVIframe)
    else:
    	print "NO"
        NDVIframeG = NDVIframe

    return NDVIframe, NDVIframeG

#loads to recorded images that have to be aligned by the user
# they will be saved in the folder cropped, that has to be created before
def load2ImagesAndCrop(dire, name1,name2, h):
	img1 = cv2.imread(dire+name1)
	img2 = cv2.imread(dire+name2)

	# mark the images as global variables to work with the listener functions
	global left
	global right

	# copy the original images to draw in them
	left = np.copy(img1)
	right = np.copy(img2)

	b,g,RED = cv2.split(img1)
	b,g,IR = cv2.split(img2)

	# get the homogrphy from the manual registration
	H = collectPairs()

	# warp the images with the homography
	cropR, cropIR, ret = warpAndCropImages(RED,IR,H)

	cv2.imshow("RGB",cropR)
	cv2.imshow("IR",cropIR)

	#write the cropped images to the filesystem
	dirname = dire + "\cropped\\"  
	cv2.imwrite(dirname + name1,cropR)
	cv2.imwrite(dirname + name2,cropIR)


# loads two already aligned images to calculate their NDVI
def load2CroppedAndNDVI(dire, name1,name2):

	img1 = cv2.imread(dire+name1)
	img2 = cv2.imread(dire+name2)	

	b,g,RED = cv2.split(img1)
	b,g,IR = cv2.split(img2)

	cv2.imshow("RGB",RED)
	cv2.imshow("IR",IR)

	# calculate the ndvi and get the color graded version
	ndvi1, ndvi2 = calculateNDVI(RED,IR,"BGR")

	# scale the NDVI from [-1;1] to [0;1] for propper display
	ndvi1 = cv2.add(ndvi1, 1.0)
	ndvi1 = cv2.divide(ndvi1, 2.0)
	
	# show the results
	cv2.imshow("NDVI",ndvi1)
	cv2.imshow("NDVI2",ndvi2)
	cv2.waitKey(0)

	#transpose to [0;255] to save as 8bit
	ndvi1 = cv2.multiply(ndvi1, 255)

	ndvi2 = cv2.multiply(ndvi2, np.array([255.0,255.0,255.0,0.0]))

	# convert back to 8bit
	intNDVI = ndvi1.astype(np.uint8)
	intNDVI2 = ndvi2.astype(np.uint8)

	# write to disc
	cv2.imwrite(dire+"\\" + name1[:-7]+"NDVI.jpg",intNDVI)
	cv2.imwrite(dire+"\\" + name1[:-7]+"NDVI3.jpg",intNDVI2)


	
# sets up the four windows to fit the screen
def setupWindows():
	SCREENWIDTH = GetSystemMetrics (0)
	SCREENHEIGHT = GetSystemMetrics (1)

	cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("RGB", SCREENWIDTH/2, (SCREENWIDTH/2)/16*9)
	cv2.moveWindow("RGB", 0, 0) 

	cv2.namedWindow("IR", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("IR", SCREENWIDTH/2, (SCREENWIDTH/2)/16*9)
	cv2.moveWindow("IR", SCREENWIDTH/2, 0) 

	cv2.namedWindow("NDVI", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("NDVI", SCREENWIDTH/2, (SCREENWIDTH/2)/16*9)
	cv2.moveWindow("NDVI", 0, (SCREENWIDTH/2)/16*9)

	cv2.namedWindow("NDVI2", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("NDVI2", SCREENWIDTH/2, (SCREENWIDTH/2)/16*9)
	cv2.moveWindow("NDVI2", SCREENWIDTH/2, (SCREENWIDTH/2)/16*9)


# global variables
RGB = False
POINTS = 0

RGBpoints = []
IRpoints = []

left = None
right = None




############################
#### MAIN STARTING POINT ###
############################


setupWindows()

dirname = os.getcwd()

# choose a directory with the samples
dirname += "\M3\\"

# namelist for iteration

namelist = ["1","2","3","o1","o2","f"]

for n in namelist[2]:
	# image names have to end with "_rgb.jpg" and "_ir.jpg"
	redName      = n + "_rgb.jpg"
	infraredName = n + "_ir.jpg"

	#reset variables
	RGB = False
	POINTS = 0
	RGBpoints = []
	IRpoints = []
	left = None
	right = None

	# choose to either crop images or 
	load2ImagesAndCrop(dirname, redName,infraredName, None)
	# calculate the ndvi
	# load2CroppedAndNDVI(dirname+"cropped\\", redName,infraredName) 
