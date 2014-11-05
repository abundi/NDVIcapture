import numpy as np
import cv2
from matplotlib import pyplot as plt

capRGB = cv2.VideoCapture(0)
try:
	capIR = cv2.VideoCapture(1)
except:
	print "no second cam"

if capRGB.isOpened() == True: # && capIR.isOpended() == True
	print "OPEN"
else:
	capRGB.open()
	# capIR.open()
	if capRGB.isOpened() == True: # && capIR.isOpended() == True
		print "now its's OPEN"
	else:
		print "NOPE BITCH"
		capRGB.release()
		# capIR.release()
		cv2.destroyAllWindows()
		exit()


capture = True

while(True):
    # Capture frame-by-frame
    if capture == True: 
    	# try:
	    ret, RGBframe = capRGB.read()
    	# except:
    	# 	print "could not read image: "


	    b,g,r = cv2.split(RGBframe)

	    # Display the resulting frame
		# bi = cv2.bilateralFilter(r, 5, 100, 5)
		# cv2.Smooth(r, r, smoothtype=CV_GAUSSIAN, param1=3, param2=0, param3=0, param4=0)

	    cv2.imshow('RED',r)
	    cv2.imshow('BLUE',b)

    if cv2.waitKey(1) & 0xFF == 32: #space: resume/pause recording
        capture = not capture

    if cv2.waitKey(1) & 0xFF == ord('s'):
    	# screenshot = RGBframe
		a = cv2.imwrite('shot.jpg',RGBframe);

	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	print "q"
	


# When everything done, release the capture
capRGB.release()
cv2.destroyAllWindows()