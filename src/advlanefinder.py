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
        self.persInvTransM = None
        self.leftFit = None     #Polygon for left lane
        self.rightFit = None    #polygon for right lane
        self.leftCurverad = None
        self.rightCurverad = None
        self.carPos = None      #carPos in meters, negative means left to middle of lane
        self.noRecoveries = 0   #no of recoveries (fetching histogram again)
        self.counter = 0
     
    

    #static: display an image
    def showImage(img, title="image"):
        cv2.imshow(title,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def writeImage(img, filename="R:\output.jpg"):
        cv2.imwrite(filename, img)

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
        AdvancedLaneFinder.writeImage(img, "R:/Distorted.jpg")
        cv2.imshow("Undistorted", undiImg)
        AdvancedLaneFinder.writeImage(undiImg, "R:/Undistorted.jpg")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def undistortImage(self, image):
        return cv2.undistort(image, self.camMatrix, self.distCoeff, None, self.camMatrix)     
        
    def initializeTransformMatrix(self, image = None):
#         src = np.float32(
#             [[575,460],
#              [704,460],
#              [1046,672],
#              [259,672]])
        src = np.float32(
            [[577,461],
             [707,461],
             [1029,666],
             [275,666]])
#         src = np.float32(
#             [[562,461],
#              [720,461],
#              [1029,666],
#              [275,666]])
        #TAKE CARE:
        #In order to calculate a valid curvature radius, I must not
        #change the dimensions in X-direction! Otherwise the curvature 
        #calculation will mess up - this is based on some constants
        #which reflect distance-meter in units of pixel.
        #Please note: it is uncritical for y direction - we always
        #assume a lenght of 720 px
        dst = np.float32(
            [[275, 30],
             [1029,30],
             [1029,720],
             [275,720]])
        self.persTransM = cv2.getPerspectiveTransform(src, dst)
        self.persInvTransM = cv2.getPerspectiveTransform(dst, src)
        
        if not image is None:
            cv2.circle(image, tuple(src[0]), 5, (255,255,255),4)
            cv2.circle(image, tuple(src[1]), 5, (255,255,255),4)
            cv2.circle(image, tuple(src[2]), 5, (255,255,0),4)
            cv2.circle(image, tuple(src[3]), 5, (255,255,0),4)
        
        return image
                
    def transformPerspective(self, image, forward = True):
        dim = image.shape[:2]
        if True == forward:
            return cv2.warpPerspective(image, self.persTransM, (dim[1], dim[0]))
        return cv2.warpPerspective(image, self.persInvTransM, (dim[1], dim[0]))
        
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
        output[ #((absgraddir >= directionThresh[0]) & (absgraddir <= directionThresh[1]) ) &\
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
            
#             print (objp, "ObjPtr", objpoints)
            
            
            if True == ret :
                #success, store the image-/object points
                objpoints.append(objp)
                imgpoints.append(corners)
                img = cv2.drawChessboardCorners(image, (9,6), corners, ret)
                cv2.imshow("Chessboard", img)
                cv2.waitKey(500)
                

        #after iterating through all our camera_cal images, we do the calibration
        print ("Calibration using {0:d} Objectpoints".format(len(objpoints)))
        print (objpoints[0].shape, imgpoints[0].shape)
        if(len(objpoints)==len(imgpoints) and 0!=len(objpoints) ):
            ret, self.camMatrix, self.distCoeff, self.rotVec, self.traVec = \
                cv2.calibrateCamera(objpoints, imgpoints, camScope, None, None)
        
        
        return self.camMatrix is not None 
    
    def findEdges(self, image):
#         if(self.counter < 100):
#             cv2.imwrite("R:\\pic{:03d}.jpg".format(self.counter), image)
#         self.counter += 1
        #val = self.applyGradient(self.applyColorSpaceTransformation(image), (40,74), (20, 100) ) 
        val = self.applyColorSpaceTransformation(image, 120)
        val2 = self.applyGradient(image, (40,74), (20, 100))
#         val2 = self.applyGradient(image, (70,110), (0, 255))
#         cv2.imshow("Input", image)
#         cv2.imshow("Color", val)
#         cv2.imshow("Gra", val2)
        val3 = np.zeros_like(val2)
        val3[(val == 255) | (val2 == 255) ] = 255
        
        return val3
#         return np.dstack( (val3, val3*0, val3*0) )
    
    
    def findPolygons(self, image):
        
        out_img = np.dstack((image*0, image*0, image*0))
        #identify the elements in the image, which aren't zero - meaning not black
        #this is done by image.nonzero!!
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Set the width of the windows +/- margin
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 50
        
        
        leftx = None
        lefty = None 
        
        if( (self.leftFit == None) | (self.rightFit == None)):
            #create historgram along x-axis from the lower half of the picture
            #to find an appropriate startpoint for lane detection
            histogram = np.sum(image[np.int(image.shape[0]/2):,:], axis = 0)
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #         cv2.circle(out_img, (leftx_base, int(image.shape[0]/2)), 5, (255,0,0),4)
    #         cv2.circle(out_img, (rightx_base, int(image.shape[0]/2)+50), 5, (255,0,0),4)
    #         cv2.circle(out_img, (midpoint, int(image.shape[0]/2)+100), 5, (255,0,0),4)
            
            noWindows = 9
            window_height = np.int(image.shape[0]/noWindows)
            
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []
            # Step through the windows one by one
            for window in range(noWindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = image.shape[0] - (window+1)*window_height
                win_y_high = image.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
                # Identify the nonzero pixels in x and y within the window
                #big matrix which contains true/false and this is converted into a matrix of indicies (nonzero)
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                
            # Concatenate the arrays of indices (from list to array)
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            
            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 
        
            # Fit a second order polynomial to each
            self.leftFit = np.polyfit(lefty, leftx, 2)
            self.rightFit = np.polyfit(righty, rightx, 2)
        else:
            left_lane_inds = ((nonzerox > (self.leftFit[0]*(nonzeroy**2) + self.leftFit[1]*nonzeroy + self.leftFit[2] - margin)) & (nonzerox < (self.leftFit[0]*(nonzeroy**2) + self.leftFit[1]*nonzeroy + self.leftFit[2] + margin))) 
            right_lane_inds = ((nonzerox > (self.rightFit[0]*(nonzeroy**2) + self.rightFit[1]*nonzeroy + self.rightFit[2] - margin)) & (nonzerox < (self.rightFit[0]*(nonzeroy**2) + self.rightFit[1]*nonzeroy + self.rightFit[2] + margin)))  
            
            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            self.leftFit = np.polyfit(lefty, leftx, 2)
            self.rightFit = np.polyfit(righty, rightx, 2)
            
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        left_fitx = self.leftFit[0]*ploty**2 + self.leftFit[1]*ploty + self.leftFit[2]
        right_fitx = self.rightFit[0]*ploty**2 + self.rightFit[1]*ploty + self.rightFit[2]
        
        ptsl = np.column_stack( (left_fitx.astype(np.int32), ploty.astype(np.int32)))
        ptsr = np.column_stack( (right_fitx.astype(np.int32), ploty.astype(np.int32)))
        pts = np.concatenate((ptsl, np.flipud(ptsr)))
        cv2.polylines(out_img, [ptsl], False, (0,0,255), 8)
        cv2.polylines(out_img, [ptsr], False, (0,0,255), 8)
        cv2.fillPoly(out_img, [pts], (0,255,0))

#         lalaLand = np.dstack((image*1, image*1, image*1))
#         cv2.imshow("lLA", lalaLand)
#         cv2.addWeighted(out_img, 0.8, lalaLand, 0.3, 0)
#         cv2.imshow("Persp", out_img)
        
        
        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        # MAKE SURE YOU DIDN'T CHANGE THE DIMENSIONS IN X-AXIS DURING
        # PERSPECTIVE TRANSFORMATION
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        
        #straight Sanity check: if the distance between the lanes 
        #at the car and in the horizont are differing in about 1m - it's time
        #to research for lanes using the histogram
        dist_betweenLanes_car = (right_fitx[-1] - left_fitx[-1]) * xm_per_pix
        dist_betweenLanes_horizont = (right_fitx[0] - left_fitx[0]) * xm_per_pix
        self.carPos = ((1280/2.) - (left_fitx[-1] + dist_betweenLanes_car/2.0))   * xm_per_pix;
        
        if(abs(dist_betweenLanes_car - dist_betweenLanes_horizont) > 1.0):
            print ("Sanity check failed distance of cams {0:.1f} {1:.1f}".format(dist_betweenLanes_car, dist_betweenLanes_horizont))
            self.leftCurverad = None
            self.rightCurverad = None
            self.leftFit = None
            self.rightFit = None
            self.noRecoveries +=1
        else:
            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
            right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
            # Calculate the new radii of curvature
            self.leftCurverad  = ((1 + (2*left_fit_cr[0 ]*y_eval*ym_per_pix + left_fit_cr[1 ])**2)**1.5) / np.absolute(2*left_fit_cr[0 ])
            self.rightCurverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
#         cv2.polylines(out_img, [ptsl], False, (0,0,255),8 )
#         cv2.polylines(out_img, [ptsr], False, (0,0,255),8 )
        
        return out_img
        
        
    def doAll(self, image):
        img = finder.undistortImage(image)
        img = finder.findEdges(img)
        img = finder.transformPerspective(img)
        img = finder.findPolygons(img)
        img = finder.transformPerspective(img, False)
        if not (self.leftCurverad is None):
            cv2.putText(img, "curvLeft {0:.2f}m curvRight {1:.2f}m".format(self.leftCurverad, self.rightCurverad), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            cv2.putText(img, "Car is {0:.1f}m from middle of lane".format(self.carPos), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            cv2.putText(img, "No recoveries {0:d}".format(self.noRecoveries), (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
        else:
            cv2.putText(img, "Sanity check failed - restart histogram search", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        return cv2.addWeighted(image, 0.8, img, 0.3, 0)
#         return img  
    
    #/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    #section declaring the static members
    #_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    #/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_            
    showImage = staticmethod(showImage)
    writeImage = staticmethod(writeImage)

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
    
#     finder.exampleShowUndistorted("../test_images/straight_lines1.jpg")
    
#     img = cv2.imread("../test_images/straight_lines2.jpg")
    img = cv2.imread("R:/pic001.jpg")
#     AdvancedLaneFinder.showImage(finder.initializeTransformMatrix(img))


    finder.initializeTransformMatrix()
#     img = finder.undistortImage(img)
#     img = finder.transformPerspective(img)
#     AdvancedLaneFinder.writeImage(img, "R:/Perspective_example001.jpg")
#     img = finder.findEdges(img)
# #     AdvancedLaneFinder.writeImage(img, "R:/FindEdges_Binary.jpg")
# #     img = finder.findPolygons(img)
# 
#     img = finder.doAll(img)
#     AdvancedLaneFinder.showImage(img)
#     exit(1)
    
    
#     AdvancedLaneFinder.showImage(finder.transformPerspective(img))
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


    clip1 = VideoFileClip("../project_video.mp4")
#     clip1 = VideoFileClip("../critical.mp4")
#     clip1 = VideoFileClip("../short.mp4")
    clipo = clip1.fl_image(lambda x: cv2.cvtColor(finder.doAll(\
              cv2.cvtColor(x, cv2.COLOR_RGB2BGR) ), cv2.COLOR_BGR2RGB ))
    clipo.write_videofile("R:\\test.mp4", audio=False)
    
#     for img in ReadFileIterator("../camera_cal"):
#         calibrateCameraUsingImage(img)
    
    pass