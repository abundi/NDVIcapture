import numpy as np
import cv2
import datetime
import os
from matplotlib import pyplot as plt
from win32api import GetSystemMetrics


def recordingLoop():
	while(True):
		# Capture frame-by-frame
		# try:
		ret, RGBframe = capRGB.read()
		ret, IRframe = capIR.read()

		# b,g,IRframe = cv2.split(IRframe)
		# b,g,RGBframe = cv2.split(RGBframe)

		# except:
		# 	print "could not read image: "


		# Filter the resulting frame
		# bi = cv2.bilateralFilter(r, 5, 100, 5)
		# cv2.Smooth(r, r, smoothtype=CV_GAUSSIAN, param1=3, param2=0, param3=0, param4=0)

		cv2.imshow('RGB',RGBframe)
		cv2.imshow('IR',IRframe)

		if cv2.waitKey(1) & 0xFF == 32: #space: resume/pause recording
			t = datetime.datetime.now()
			dirname = "/shots"
			directory = os.getcwd()+"\\M3"
			# directory = os.path.dirname(dirname)
			print directory
			if not os.path.exists(directory): 
				print "does not"
				os.makedirs(directory)

			filename = str(t.day) + "_" + str(t.month) + "_" + str(t.year) + "_-_" + str(t.hour) + "_" + str(t.minute) + "_" + str(t.second) + "_#_"
			a = cv2.imwrite(directory+"/"+filename+'RGB.jpg',RGBframe)
			b = cv2.imwrite(directory+"/"+filename+'IR.jpg' ,IRframe)
			cv2.destroyAllWindows()
			


		if cv2.waitKey(1) & 0xFF == ord('q'):
			capIR.release()
			capRGB.release()
			cv2.destroyAllWindows()
			exit()	


 	
capRGB = cv2.VideoCapture(0)
try:
	capIR = cv2.VideoCapture(1)
except:
	print "no second cam"

if capRGB.isOpened() == True: # && capIR.isOpended() == True
	print "OPEN"
else:
	capIR.open()
	if capRGB.isOpened() == True: # && capIR.isOpended() == True
		print "now its's OPEN"
	else:
		print "NOPE BITCH"
		capRGB.release()
		capIR.release()
		cv2.destroyAllWindows()
		exit()

SCREENWIDTH = GetSystemMetrics (0)
SCREENHEIGHT = GetSystemMetrics (1)

cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RGB", SCREENWIDTH/2, (SCREENWIDTH/2)/16*9)
cv2.moveWindow("RGB", 0, 0) 

cv2.namedWindow("IR", cv2.WINDOW_NORMAL)
cv2.resizeWindow("IR", SCREENWIDTH/2, (SCREENWIDTH/2)/16*9)
cv2.moveWindow("IR", SCREENWIDTH/2, 0) 

capRGB.set(cv2.CAP_PROP_FPS, 30)
capRGB.set(3,1280)
capRGB.set(4,720)

capIR.set(cv2.CAP_PROP_FPS, 30)
capIR.set(3,1280)
capIR.set(4,720)
recordingLoop()


