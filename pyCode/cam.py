import numpy as np
import cv2
import os
import datetime
from matplotlib import pyplot as plt
from win32api import GetSystemMetrics

def colorGradeBG(ndvi):


    red = np.zeros(ndvi.shape,np.float32)
    # gradedNDVI = np.zeros((1280,720,3),np.uint8)

    blue = cv2.subtract(1.0, ndvi)
    # blue = cv2.multiply(ndvi,255)

    # print blue.shape

    # green = cv2.multiply(ndvi,255) 

    # print green.dtype

    gradedNDVI = cv2.merge((blue,ndvi,red))

    # return cv2.mixChannels(blue ,gradedNDVI,[0,1])
    # return cv2.mixChannels(green,gradedNDVI,[0,2])
    return gradedNDVI
    # return (255.0 * (1 - ndvi),255.0 * ndvi, 0)

def colorGradeBGR(ndvi):
    if ndvi < 0.5:
        return (511.0 * (0.5 - ndvi),511.0 * ndvi, 0)
    else:   
        return (0, 511.0 * (1 - ndvi), 511.0 * (ndvi - 0.5))


def autoAlignFrames(grayImage1, grayImage2):

    WIDTH = grayImage1.shape[1] 
    HEIGHT = grayImage1.shape[0] 
    IMGSIZE = (WIDTH, HEIGHT)

    print grayImage1.dtype
    print grayImage1.shape
    print grayImage2.dtype
    print grayImage2.shape

    ########### IMAGE REGISTRATION ###########
    # get Keypoints and Descriptors
    keypoints1, des1 = orb.detectAndCompute(grayImage1,None)
    keypoints2, des2 = orb.detectAndCompute(grayImage2,None)

    # Match them
    matches = bf.match(des1,des2)

    # Sort them
    sortedKeyPoints1 = np.empty([len(matches),2])
    sortedKeyPoints2 = np.empty([len(matches),2])

    for i in range (0,len(matches)):
       sortedKeyPoints1[i]=keypoints1[matches[i].queryIdx].pt;
       sortedKeyPoints2[i]=keypoints2[matches[i].trainIdx].pt;
    

    # Find Homography
    H, mask = cv2.findHomography( sortedKeyPoints1, sortedKeyPoints2, cv2.RANSAC );
 
    if H is None:
        print "No Homography found"
        return grayImage1, grayImage2, False

    print len(matches)
    print H
    # homographys.append(H);

    # Warp the IR image on the RED image
    warpedGray1 = np.zeros(grayImage1.shape)
    warpedGray1 = cv2.warpPerspective(grayImage1, H, IMGSIZE, warpedGray1, cv2.INTER_LINEAR)


    print "warped"
    ########## IMAGE TRANSFORMATION ##########

    # Calculate biggest rectangle
    gray1Corners = np.zeros([4,2], dtype="float32")

    gray1Corners[0] = [0.0  ,0.0   ]                    # 0 top Left
    gray1Corners[1] = [0.0  ,HEIGHT]        # 1 bot Left
    gray1Corners[2] = [WIDTH,0.0   ]                    # 2 top Right
    gray1Corners[3] = [WIDTH,HEIGHT]        # 3 bot Right

    # Transform IR Corners in Projected Space
    gray1Corners = np.array([gray1Corners])
    gray1WarpedCorers = cv2.perspectiveTransform(gray1Corners, H)

    # Get Bounding Box
    boundLeft  = max(0     ,gray1WarpedCorers[0][0][0], gray1WarpedCorers[0][1][0])
    boundRight = min(WIDTH ,gray1WarpedCorers[0][2][0], gray1WarpedCorers[0][3][0])
    boundTop   = max(0     ,gray1WarpedCorers[0][0][1], gray1WarpedCorers[0][2][1])
    boundBot   = min(HEIGHT,gray1WarpedCorers[0][1][1], gray1WarpedCorers[0][3][1])

    topLeft = (boundLeft, boundTop)
    botRight = (boundRight, boundBot)

    ROI = (topLeft, botRight)

    # Crop the input Images
    croppedGray1 = warpedGray1[boundTop:boundBot, boundLeft:boundRight]
    croppedGray2 = grayImage2[boundTop:boundBot, boundLeft:boundRight]

    # Blend Images for visualisation
    # weightedIMG = cv2.addWeighted( warpedIR, 0.5, grayImage2, 0.5, 0.0)
    # croppedWeighted = weightedIMG[boundTop:boundBot, boundLeft:boundRight]


    # cv2.imshow("blended",weightedIMG)
    # cv2.imshow("cropped",croppedWeighted)

    return croppedGray1, croppedGray2, True



# Frames have to be aligned
def calculateNDVI(REDframe, IRframe, grading ):
    # TODO calcStuff

    # NDVIframe = np.zeros(IRframe.shape,np.uint8)
    # NDVIframeG = np.zeros((1280,720,3),np.uint8)

    fRED = REDframe.astype(np.float32)
    fRED = cv2.divide(fRED, 255.0)
    fIR = IRframe.astype(np.float32)
    fIR = cv2.divide(fIR, 255.0)


    numerator = cv2.subtract(fIR, fRED)
    denumerator = cv2.add(fIR, fRED)
    fraction = cv2.divide(numerator, denumerator)

    print REDframe.shape
    print cv2.mean(fIR)
    # print cv2.mean()

    NDVIframe = fraction
    # NDVIframe = cv2.multiply(fraction, 255)

    if grading is "BG":
        NDVIframeG = colorGradeBG(NDVIframe)
    if grading is "BGR":
        NDVIframeG = colorGradeBGR(RNDVIframe)


# for x in range (0,ir.shape[0]):
#     for y in range (0,ir.shape[1]):
#         # ndvi[x,y] =  ir[x,y] - r[x,y]
#         numerator   = float(ir[x,y]) / 255.0 - float(r[x,y]) / 255.0
#         denominator = float(ir[x,y]) / 255.0 + float(r[x,y]) / 255.0
#         if (denominator is 0.0):
#             fraction = 0.0
#             print "division by zero exception"
#         else:
#             fraction  = (numerator / denominator) / 2.0 + 0.5
#         ndvi[x,y] =  int(fraction * 255.0) 
#         ndviRGB[x,y] =  colorGradeBGR(fraction) 

#         # ndvi[x,y] =  int(float(ir[x,y] - r[x,y]) / float(ir[x,y] + r[x,y]) * 255.0 )


#     NDVIframe = REDframe

    return NDVIframe, NDVIframeG

# END calculateNDVI 


def recordingLoop():
    # (ret, RGBframe) = capRGB.read()

    while(recording):
        ret, RGBframe = capRGB.read()
        ret2, IRframe = capIR.read()


        #  single cam
        # IRframe = RGBframe
        # (ret, RGBframe ) = capRGB.read()

        cv2.imshow('RGB',RGBframe)
        cv2.imshow('IR',IRframe)

        if cv2.waitKey(1) & 0xFF is 32: #space: resume/pause recording
            # cv2.destroyAllWindows()
            showImage(RGBframe,IRframe);
            break;

        if cv2.waitKey(1) & 0xFF is ord('q'):
            capIR.release()
            capRGB.release()
            cv2.destroyAllWindows()
            exit()    

#  END Recording Loop


def showImage(RGBframe,IRframe):

    # split channels
    b,g,REDchannel = cv2.split(RGBframe)
    b,g,IRchannel = cv2.split(IRframe)

    cv2.imshow('RGB',REDchannel)
    cv2.imshow('IR',IRchannel)

   
    t = datetime.datetime.now()
    wdir = os.getcwd() 
    dirname = str(t.day) + "_" + str(t.month) + "_" + str(t.year) + "_-_" + str(t.hour) + "_" + str(t.minute) + "_" + str(t.second) + "__"
    dirname = wdir+ "\\" + dirname
    os.makedirs(dirname)
    print "created dir: " + dirname
    cv2.imwrite(dirname+"\RGBframe.jpg",RGBframe)
    cv2.imwrite(dirname+"\IRframe.jpg",IRframe)
    cv2.imwrite(dirname+"\REDchannel.jpg",REDchannel)
    cv2.imwrite(dirname+"\IRchannel.jpg",IRchannel)



    # Filter the resulting frame
    # bi = cv2.bilateralFilter(r, 5, 100, 5)
    # cv2.Smooth(r, r, smoothtype=CV_GAUSSIAN, param1=3, param2=0, param3=0, param4=0)

    # align images
    alignedRED, alignedIR, success = autoAlignFrames(REDchannel,IRchannel)

    if not success:
        recording = True
        print "sorry didnt work"
        # cv2.destroyWindow("NDVI")
        recordingLoop()

    #write all the images to disc
    cv2.imwrite(dirname+"REDaligned.jpg",alignedRED)
    cv2.imwrite(dirname+"IRaligned.jpg",alignedIR)

    # calc NDVI
    NDVIframe, gradedNDVIframe = calculateNDVI(alignedRED, alignedIR, "BG")

    # show results
    cv2.imshow('NDVI', NDVIframe)
    cv2.imshow('NDVI2', gradedNDVIframe)

    cv2.imwrite(dirname+"NDVI.jpg",NDVIframe)
    cv2.imwrite(dirname+"NDVI2.jpg",gradedNDVIframe)


    if cv2.waitKey(1) & 0xFF is ord('q'):
        capIR.release()
        capRGB.release()
        cv2.destroyAllWindows()
        exit()    

    if cv2.waitKey(0) & 0xFF is 32:
        
        recording = True
        # cv2.destroyWindow("NDVI")
        recordingLoop()

    cv2.waitKey(1)

#start

capRGB = cv2.VideoCapture(0)
try:
    capIR = cv2.VideoCapture(1)
except:
    print "no second cam"

if (capRGB.isOpened() is True) and (capIR.isOpened() is True):
    print "both OPEN"
else:
    capIR.open()
    if (capRGB.isOpened() is True):  # && capIR.isOpended() is True
        print "now its's OPEN"
    else:
        print "NOPE doesnt work bro"
        capRGB.release()
        capIR.release()
        cv2.destroyAllWindows()
        exit()



# check for the other cam
# check raw data
capRGB.set(cv2.CAP_PROP_FPS, 30)
capRGB.set(3,1280)
capRGB.set(4,720)

capIR.set(cv2.CAP_PROP_FPS, 30)
capIR.set(3,1280)
capIR.set(4,720)


orb = cv2.ORB()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

recording = True

# for i in range(0,18):
#   print capRGB.get(i)
#

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

recordingLoop()






# When everything done, release the capture
capRGB.release()
cv2.destroyAllWindows()