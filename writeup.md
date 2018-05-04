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

[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the [camera_calibration.py] file in
the function [`Model.calibrate()`]. First I [load all calibration images].
[Prepare an array of object points] of the chessboard corners in the real
world. Then I [prepare an array of image points] associated with the object
points of the chessboard corners by calling `cv2.findChessboardCorners()` on
each of the calibration images. Then I [calculate the camera matrix] and the
distortion coefficients by calling `cv2.calibrateCamera()` with the
prepared object and image points. If you run the `camera_calibration.py`
script, it will perform calibration an [save the matrix and coefficients]
into the [calibration data file], which [can be loaded] in the later stages
of the detection pipeline, saving time consumed by recalibration of camera.

I applied this distortion correction to one of the calibration images,

![Original][original_calibration_image]

by calling [`Model.undistort()`] and obtained the following result:

![Undistorted][undistorted_calibration_image]

[camera_calibration.py]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/master/camera_calibration.py
[`Model.calibrate()`]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/1c6a48ae76ba7666e9e05fc392e695cc09fb5a1a/camera_calibration.py#L8
[load all calibration images]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/1c6a48ae76ba7666e9e05fc392e695cc09fb5a1a/camera_calibration.py#L12
[Prepare an array of object points]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/162c65f2af7691da8a5975d05c4ee271e2e3ccf7/camera_calibration.py#L16
[prepare an array of image points]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/162c65f2af7691da8a5975d05c4ee271e2e3ccf7/camera_calibration.py#L21
[calculate the camera matrix]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/162c65f2af7691da8a5975d05c4ee271e2e3ccf7/camera_calibration.py#L29
[original_calibration_image]: ./examples/original_calibration_image.jpg "Original Calibration Image"
[undistorted_calibration_image]: ./examples/undistorted_calibration_image.jpg "Undistorted Calibration Image"
[`Model.undistort()`]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/b46d739a75e7cce0a7e338a3033aa780ccd0c16e/camera_calibration.py#L39
[save the matrix and coefficients]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/b46d739a75e7cce0a7e338a3033aa780ccd0c16e/camera_calibration.py#L43
[calibration data file]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/master/camera_model.npz
[can be loaded]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/b46d739a75e7cce0a7e338a3033aa780ccd0c16e/camera_calibration.py#L49

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here is the demonstration of the distortion correction (original and
undistorted images):

![Original Test Image][original_test_image]
![Undistorted Test Image][undistorted_test_image]

[original_test_image]: ./examples/original_test.jpg
[undistorted_test_image]: ./examples/undistorted_test.jpg

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
