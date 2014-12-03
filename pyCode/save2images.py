import numpy as np
import cv2
import datetime
import os
from matplotlib import pyplot as plt
from win32api import GetSystemMetrics


def recordingLoop():
	while(True):
		# Capture frame-by-frame
		ret, RGBframe = capRGB.read()
		ret, IRframe = capIR.read()

		# show the captured frames
		cv2.imshow('RGB',RGBframe)
		cv2.imshow('IR',IRframe)

		 #space: resume/pause recording
		if cv2.waitKey(1) & 0xFF == 32:
			t = datetime.datetime.now()
			dirname = "/shots"
			directory = os.getcwd()+"\\M3"
			if not os.path.exists(directory): 
				print "does not"
				os.makedirs(directory)

			filename = str(t.day) + "_" + str(t.month) + "_" + str(t.year) + "_-_" + str(t.hour) + "_" + str(t.minute) + "_" + str(t.second) + "_#_"
			a = cv2.imwrite(directory+"/"+filename+'RGB.jpg',RGBframe)
			b = cv2.imwrite(directory+"/"+filename+'IR.jpg' ,IRframe)

		# close on key q
		if cv2.waitKey(1) & 0xFF == ord('q'):
			capIR.release()
			capRGB.release()
			cv2.destroyAllWindows()
			exit()	



############################
#### MAIN STARTING POINT ###
############################

#try get both video captures
capRGB = cv2.VideoCapture(0)
try:
	capIR = cv2.VideoCapture(1)
except:
	print "no second cam found"
	exit()

if capRGB.isOpened() == True: # && capIR.isOpended() == True
	print "FIRST OPEN"
else:
	capIR.open()
	if capRGB.isOpened() == True: # && capIR.isOpended() == True
		print "now its's OPEN"
	else:
		print "NOT OPEN"
		capRGB.release()
		capIR.release()
		cv2.destroyAllWindows()
		exit()


# setup the windows
SCREENWIDTH = GetSystemMetrics (0)
SCREENHEIGHT = GetSystemMetrics (1)

cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RGB", SCREENWIDTH/2, (SCREENWIDTH/2)/16*9)
cv2.moveWindow("RGB", 0, 0) 

cv2.namedWindow("IR", cv2.WINDOW_NORMAL)
cv2.resizeWindow("IR", SCREENWIDTH/2, (SCREENWIDTH/2)/16*9)
cv2.moveWindow("IR", SCREENWIDTH/2, 0) 

# setup the cameras
capRGB.set(cv2.CAP_PROP_FPS, 30)
capRGB.set(3,1280)
capRGB.set(4,720)

capIR.set(cv2.CAP_PROP_FPS, 30)
capIR.set(3,1280)
capIR.set(4,720)

# start recording
recordingLoop()


