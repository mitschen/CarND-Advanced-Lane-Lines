'''
Created on 08.07.2017

@author:    michael scharf
@email:     mitschen@gmail.com
@git:       www.github.com/mitschen  
'''

#TODO:
# 1. compute cam calibration matrix and distortion coefficients
# 2. use color transformation
# 3. use gradients
# 4. get binary image 
# 5. apply perspective transofmration
# 6. detect lane pixels
# 7. determine curvature of the line
# 8. apply lane boundary to the original image
# 9. output 


import numpy as np
import cv2
import os
import glob
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
#enable to download ffmpeg.win32.exe
# import imageio


#Simple iterator used to read file-by-file using cv2.imread
class ReadFileIterator(object):
    def __init__(self, filepath, fileEnding = '*.jpg'):
        self.filenames = glob.glob(os.path.join(filepath, fileEnding))
        self.index = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if(self.index < len(self.filenames)):
            i = self.index
            self.index+=1 
            return cv2.imread(self.filenames[i])
        else :
            raise StopIteration()

class AdvancedLaneFinder(object):
#     def readImages(pathToImage, fileEnding = "*.jpg"):
#         filenames = glob.glob(os.path.join(pathToImage, fileEnding))
#         print (filenames)
     
    def __init__(self):
        #Members used for undistortion
        self.camMatrix = None   #CamMatrix
        self.distCoeff = None   #Distortion coefficient
        self.rotVec = None      #rotation vectors
        self.traVec = None      #translation vectors
        self.persTransM = None
        self.counter = 0
    

    #static: display an image
    def showImage(img, title="image"):
        cv2.imshow(title,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def writeMembers(self, path):
        dataToWrite = (self.camMatrix, self.distCoeff, self.rotVec, self.traVec)
        with open(path, 'wb') as file:
            pickle.dump(dataToWrite, file)
            print ("Write persistency to ", path)
    
    def readMembers(self, path):
        with open(path, 'rb') as file:
            dataToRead = pickle.load(file)
            self.camMatrix, self.distCoeff, self.rotVec, self.traVec = dataToRead
            print ("Read persistency to ", path)
            
        
    #display an example normal/ undistorted
    def exampleShowUndistorted(self, filepath):
        img = cv2.imread(filepath)
        undiImg = cv2.undistort(img, self.camMatrix, self.distCoeff, None, self.camMatrix)        
        cv2.imshow("Distorted", img)
        cv2.imshow("Undistorted", undiImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def initializeTransformMatrix(self, image = None):
        src = np.float32(
            [[575,460],
             [704,460],
             [1046,672],
             [259,672]])
#         dst = np.float32(
#             [[0,0],
#              [1050-245,0],
#              [1050-245,680-460],
#              [0,680-460]])
        dst = np.float32(
            [[100, 30],
             [1180,30],
             [1180,690],
             [100,690]])
        self.persTransM = cv2.getPerspectiveTransform(src, dst)
        
        if not image is None:
            cv2.circle(image, tuple(src[0]), 5, (255,255,255),4)
            cv2.circle(image, tuple(src[1]), 5, (255,255,255),4)
            cv2.circle(image, tuple(src[2]), 5, (255,255,0),4)
            cv2.circle(image, tuple(src[3]), 5, (255,255,0),4)
        
        return image
                
    def transformPerspective(self, image):
        dim = image.shape[:2]
        return cv2.warpPerspective(image, self.persTransM, (dim[1], dim[0]))
        
    def applyColorSpaceTransformation(self, image, thresh = 150):
        sat = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,2]
        red = image[:,:,2]
        output = np.zeros_like(sat)
        output[(sat>=thresh) & (red >= thresh)] = 255
        return output
        

    def applyGradient(self, image, directionThresh, xgradThresh):
        img = image
        if(len(image.shape) ==3):
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #apply sobel operator in X direction
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=9)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=9)
        
        #calc the absolute gradiant direction
        directionThresh = (directionThresh[0] * np.pi / 180.,directionThresh[1] * np.pi / 180.) 
        absgraddir = np.arctan2(np.absolute(sobely),np.absolute(sobelx))
        
        #take the absolute of the sobel
        absSobel = np.absolute(sobelx)
        #convert to 8 bit gray-value
        scaledSobel = np.uint8(255*absSobel/np.max(absSobel))
        output = np.zeros_like(scaledSobel)
        output[ ((absgraddir >= directionThresh[0]) & (absgraddir <= directionThresh[1]) ) &\
               ((scaledSobel >= xgradThresh[0]) & (scaledSobel <= xgradThresh[1]) )] = 255
        return output

    def calibrateCameraUsingImage(self, imagepath):
        #member to store image and object points of each camera
        objpoints = []
        imgpoints = []
        #we're expecting a grid of 9,6 corners
        #TODO: it is not working for a shape of 9,5!!! What's the reason for that?
        noCorners = (9,6)
        
        #object points - 3D points are simply numbered from 0 to x/y
        #this var is used as dummy for the objectpoints
        objp = np.zeros( ((noCorners[0]*noCorners[1]), 3), np.float32)
        #align only the first two columns to the grid
        objp[:,:2] = np.mgrid[0:noCorners[0], 0:noCorners[1]].T.reshape(-1,2)

        #CamScope will store the hight and width of the camera scope    
        camScope = None
        #iteratore over all images in the path
        for image in ReadFileIterator(imagepath):
            #first step, we need to convert the image into grayscale image to allow
            #easier detection of the chessfield
            
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            #store the camScope and guarantee that we're not mixing up
            #images from different cameras
            if(camScope == None):
                camScope = img.shape[::-1];
            else:
                #calibration image 15 is different scale
                #therefore i've disabled the assertion
                #assert(camScope == img.shape[::-1])
                pass
            #find the chessboard, store the image points in corners
            ret, corners = cv2.findChessboardCorners(img, noCorners, None)
            
            if True == ret :
                #success, store the image-/object points
                objpoints.append(objp)
                imgpoints.append(corners)
                image = cv2.drawChessboardCorners(image, (9,6), corners, ret)
                cv2.imshow('img',image)
                cv2.waitKey(500)

        #after iterating through all our camera_cal images, we do the calibration
        if(len(objpoints)==len(imgpoints) and 0!=len(objpoints) ):
            ret, self.camMatrix, self.distCoeff, self.rotVec, self.traVec = \
                cv2.calibrateCamera(objpoints, imgpoints, camScope, None, None)
        
        
        return self.camMatrix is not None 
    
    def findEdges(self, image):
#         if(self.counter < 100):
#             cv2.imwrite("R:\\pic{:03d}.jpg".format(self.counter), image)
#         self.counter += 1
        #val = self.applyGradient(self.applyColorSpaceTransformation(image), (40,74), (20, 100) ) 
        val = self.applyColorSpaceTransformation(image, 170)
        val2 = self.applyGradient(image, (40,74), (20, 100))
        val3 = np.zeros_like(val2)
        val3[(val == 255) | (val2 == 255) ] = 255
        
        return np.dstack( (val3, val3*0, val3*0) )
    
    #/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    #section declaring the static members
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    #/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_            
    showImage = staticmethod(showImage)

if __name__ == '__main__':
    
#     AdvancedLaneFinder.readImages("../camera_cal")
    
    PERSISTENCE = "R:/pers.bin"
    finder = AdvancedLaneFinder()
    #avoid the calibration all the time from beginning
    if(True == os.path.isfile(PERSISTENCE)):
        finder.readMembers(PERSISTENCE)
    else:
        finder.calibrateCameraUsingImage("R:/camera_cal")
        finder.writeMembers(PERSISTENCE)
    
    #finder.exampleShowUndistorted("R:/camera_cal/calibration9.jpg")
    
    img = cv2.imread("../test_images/straight_lines1.jpg")
#     AdvancedLaneFinder.showImage(finder.initializeTransformMatrix(img))
    AdvancedLaneFinder.showImage(finder.findEdges(finder.transformPerspective(finder.initializeTransformMatrix(img))))
    
    
    AdvancedLaneFinder.showImage(finder.transformPerspective(img))
# #    img = cv2.imread("R://pic042.jpg")
#     val = finder.applyColorSpaceTransformation(img, 170)
#     cv2.imshow("ColorSpace", val)
#     val2 = finder.applyGradient(img, (40,74), (20, 100))
#     cv2.imshow("Gradient", val2)
#     val3 = np.zeros_like(val2)
#     val3[(val == 255) | (val2 == 255) ] = 255
#     cv2.imshow("Color&Grad",val3)
#     AdvancedLaneFinder.showImage(val)
#     AdvancedLaneFinder.showImage(finder.applyGradient(val, (40,74), (20, 100)))


    
#     imageio.plugins.ffmpeg.download()


#     clip1 = VideoFileClip("../project_video.mp4")
#     clipo = clip1.fl_image(lambda x: finder.findEdges(\
#               cv2.cvtColor(x, cv2.COLOR_RGB2BGR) ) )
#     clipo.write_videofile("R:\\test.mp4", audio=False)
    
#     for img in ReadFileIterator("../camera_cal"):
#         calibrateCameraUsingImage(img)
    
    pass