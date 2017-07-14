## Writeup Template



### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.



---



**Advanced Lane Finding Project**



The goals / steps of this project are the following:



* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

* Apply a distortion correction to raw images.

* Use color transforms, gradients, etc., to create a thresholded binary image.

* Apply a perspective transform to rectify binary image ("birds-eye view").

* Detect lane pixels and fit to find the lane boundary.

* Determine the curvature of the lane and vehicle position with respect to center.

* Warp the detected lane boundaries back onto the original image.

* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



[//]: # (Image References)



[image1]: ./mypics/Distorted.jpg "distorted chessplate"
[image2]: ./mypics/Undistorted.jpg "undistorted chessplate"
[image3]: ./mypics/Undistorted_Lanes.jpg "undistorted lanes"
[image4]: ./mypics/FindEdges.jpg "Input for finding edges"
[image5]: ./mypics/FindEdges_Binary.jpg "Result of finding edges"
[image6]: ./mypics/Perspective.jpg "Perspective transformation"
[image7]: ./mypics/PerspEdges.jpg "Edges for perspective transformation"
[image8]: ./mypics/exampleLane.jpg "projected Lanes"






[video1]: ./project_video.mp4 "Video"



## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points



### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  



---



### Writeup / README



#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  



You're reading it!



### Camera Calibration


As the first step in the advanced lane finder we do the camera calibration. The camera calibration is necessary to get rid of distortion - imagine a straight lines which get's curved because of distortion.

The calibration takes place in function `calibrateCameraUsingImage` which more or less reflects the pipe [opencv](http://docs.opencv.org/trunk/dc/dbb/tutorial_py_calibration.html) is describing. I've implemented an iterator which traverses through the calibration files stored in the subfolder `camera_cal`. The corners i'm searching for are 9 horizontally and 6 vertically. The function `cv2.findChessboardCorners` providing a set of imagepoints. As objectpoints (the real points in the scenary) i'm using a simple grid from 0-8 and 0-5. For each image i'm storing the resulting imagepoints and objectpoints in list. These two list are used at the end to calibrate the camera. The camera matrix as well as the distortion coefficients, rotation and translation vectors are stored as members of the `AdvancesLaneFinder` class - the intention behind this is, that we later want to use these members to undistort the camera images from the video.

The result of this distortion could be seen in the following pictures for which I've used the method `exampleShowUndistorted` to create both of them.
![alt text][image1]
The best idea of the distortion operation you'll get if you try to draw a straight line from one end of the chess board to the other one. As you can see in the distorted image above, the horizontal line is very much aligned with the chessboard at the border but differs from the chessboard in the middle. 

![alt text][image2]

In contrast you can see the undistorted image here - the difference between the chessboard lines and the drawn red line is still there, but it is not anylonger that big.

In other words the distortion operation still has space for improvement ;-).

A very strange thing I was facing during the camera calibration operation is, that for some calibration images the `cv2.findChessboardCorners`failed and therefore the calibration didn't take advantage of these screenshots. Furthermore I've figured out that at least one image did not align with the image height and width the other images had - this was not a big deal for calibration but caused my sanity checks to spit an assertion (see about line 200). 

The calibration of the camera takes always a few moments - in order to skip this step - the result will anyway be the same when using same input data - I've implemented two methods to store/ restore the calibration data. See `writeMembers` and `readMembers` function. 
Doing the calibration step will show each image successfully used for calibration for about half a second.

### Pipeline (single images)


The pipeline the `AdvancedLaneFinder` class is doing is more or less reflected in the `doAll` function which will step-by-step call the following methods:

1. `undistortImage` is used to remove distortion from the input image
2. `findEdges` will apply gradient and color based edge detection
3. `transformPerspective` will transform the camera image with edged to the bird view perspective
4. `findPolynomials` will try to match a order two polynom to the left and right lane, put a polygon into the image
5.  `transformPerspective` will undo the bird-view perspective
6.  and then some instructions to write curvature radius and distance from middle lane into the image using `cv2.putText` function.


#### 1. undistortImage
As mentioned, the first step is to remove distortion of the input image. This is done by using the data we've fetched via camera calibration and apply `cv2.undistort` to the image. The result you'll see in the following picture.

![alt text][image3]

The difference is very hard to see in comparism with the original picture which was *straigh_lines1.jpg*. The best way to do it anyway - as with the chessboard - is to try to put a straight line through the lanes in the right part of the image... but to be honest, i'm not sure if in this setup the distortion brings any benefit.


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To make the lane detection I'm first applying different "detection" and did the perspective transformation afterwards. The edge detection is done in method `findEdges`
As a first step I'm applying color space transformation and applying a threshhold for the saturation channel and for the red channel. The threshold is defined only for the lower-thresh because i've assumed high thresholds - meaning from dark to bright are always welcome. This is done in the method `applyColorSpaceTransformation` and returns as a result the edges in which saturation and red channel are both above the threshold. I've recognized that especially the lanes in foreground - even on bright background are very well detected. 

In a second function `applyGradient` I'm doing the Sobel-operation with a kernel of 9. As a result of this function I'm filtering only for those edges which fullfil the x-gradient as well as the direction gradient. This especially gave good results in the back of the image where the lanes are almost invisible for edges based on my color-gradient. 

![alt text][image4]
The result is a very clear lane, but many noise as well. But after applying the perspective transformation, the noise is more or less negligible due to the fact, that it is not anylonger in the scope of detection.
![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To transform the perspective of the image, I've implemented a function called `transformPerpective` which does the transformation and the function `initializeTransformMatrix` which is used to calculate the matrix to and from birds perspective. By passing an image to the `initializeTransformMatrix` an image is returned which shows the location of the transformation markers. 

In `transformPerspective` the cv2 function `warpPerspective` is used with the previously calculated transformation matrix. 

For the perspective settings I've made good experience with the following hardcoded values
```
        src = np.float32(
            [[577,461],
             [707,461],
             [1029,666],
             [275,666]])
        dst = np.float32(
            [[275, 30],
             [1029,30],
             [1029,720],
             [275,720]])
```

The white and blue circles in the image below show where these edges are located on the straight lines example image.

![alt text][image7]

As you can see i've streched the image in height but kept the dimensions in width. This transformation is a very critical point because it could lead to very strange behaviours. A typically problem I was facing was, that straigth lines didn't look straight after the transformation. This leads to very strange polynoms which then ends up in very strange curvatures. Another problem I was facing: keeping the dimension along y-axis leads to very sharp turns - this results in failures in the window-based polynomial finder. The reason for that was that the initial statingpoint (identified by mean of histogram) wasn't close enougth to the start of the lanes - so the resulting polynomial become  completely different than the one describing the lanes. 
At the end these values are identified by rule of thumb: we don't know the distance in height (meaning along y-axis) and therefore any transformation leads to a change in the curvature. But as you can see in the image - it's fine enought.

![alt text][image6]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
To find the 2nd order polynome I've started with the approach suggested in the chapter 33. The implemenation can be found in the function `findPolynomials` which combines the 
- histogram and sliding window approach to initially find the indices for points which seems to belong to the lanes
- the incremental forward alignment for subsequent pictures, not using the histogram approach
- the drawing of the polygons which should reflect the lanes
- the calculation of the curvature (based on constants factors)

The algorithm works as following: There are two branches which can entered, depending on the state of the `AdvLaneFinder` class. If we're processing the first image, we do the histogram approach. If we have already the information of the polynomials from a previous execution, we apply an incremental forward alignment.

Let's start with the incremental forward alignment (as presented in the chapter 33 of Project Advances Lane Findings) : In this case we're using the polynomials of left and right lane to specify a window in x dimension with the middle at the polynomial value for a height y and the width of `margin` in each direction. This window is used along the whole y axis to identify all indices of pixels which aren't zero. The expection is then, that these windows contains many pixels of the lane in the current frame. By using `np.polyfit` these indices are used to estimate the coefficients of the polynomial of the current frame.

In case that we start the lane finder the first time and therefor don't have a polynomial from previous frames, we apply the histogram approach: First of all a histogram of the lower half of the image is taken. The result of the histogram is seperated in a left and a right part - we assume that the car is in the middle of the lane and therefore one lane must be left of the middle, the other on the right. Taking the max of all histogram-points will lead to a very good startpoint of the lane.... but this will only the case if we've an almost straight lane. Using the startpoint, we span a window of again `margin` width in x direction and `image-height/9` in y direction and identify all indices which represent non-zero pixels in the window. Using the mean x-indice of the current window as startpoint for the next-window calculation, we travers along y axis for the whole image frame. As in the incremental forward approach, the indices are used to identify the coefficients of the polynomial using `np.polyfit`. 
In contrast to the suggested approach from the udacity term I've realized a tendency calculation in addition. The intention is to take care of very sharp turns by expanding the searchwindow in direction of the mean differences. In other words, if the mean of current frame in contrast to the mean of previous frame has moved for x pixels to the left, I've expanded the searchwindow to the left for x pixels as well.
Furthermore this tendency was used to move the window starting point to left or right in situations in which we didn't find enought pixels in the previous iterations.

At the end we're receiving the coefficients of the estimated 2nd order polynomial representing the left and the right lane.

In the second part of the function we're 
- drawing the polygon which reflects the current lane: `cv2.fillPoly`
- we calculating the curvature using some hard-coded px to meter factor: 
-- 30/720 for y and 
-- 3.7/700 for x axis
- we're doing a simply sanity check which in case will reset the polynomial and retriggers the histogram approach
-- measure the distance of left and right curvature - in case that these are larger than a certain threshold of 2 meters, we assume that at least one polynomial is not anylonger aligned with its lane
- and we identify the carposition in relation to the current lane


An example of the whole chain could be found in the following picture (please note: after the perspective transformation back from birdview):

![alt text][image8]


On top of the frame there is the 
* estimated lane hightlighted in green
* the middle of the image as a black line of about 120 px in height (= car position)
* the middle lane as a red line of about 120px in height (= middle of lane)
* left and right in thin red line the polynomial
* and some details in text


You can find the whole project video [here](https://github.com/mitschen/CarND-Advanced-Lane-Lines/blob/master/result.mp4).

---



### Discussion



#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?



Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

