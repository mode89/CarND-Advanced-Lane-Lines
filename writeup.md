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

First, I tried to use gradients to create a binary image. But I wasn't
satisfied by the quality of the output: it always seemed to be very noisy.
I decided to give it a try to a convolutional neural network. My intuition
was that the convolutional kernel facilitates almost the same goal as the
Sobel's kernel, and I would say that the Sobel filter is the special case of
a convolutional kernel. I've started with two convolution layers, where the
input layer performs filtering, and the output 1x1 convolution layer just do
regression of the filters' outputs. I think, this architecture resembles the
original approach where we filter the original image using gradient, colors
and then try to mix these filtered images. I played with different number of
kernels, kernel sizes, activation functions, and number of convolution
layers - and ended up with four convolution layers:

* 16 7x7 filters
* 12 5x5 filters
* 6  3x3 filters
* 1  1x1 filter

Every layer has 1x1 strides, same padding and ReLU activation.

I feed the network with 10 times minified bird eye view RGB images. It
outputs same sized grayscale image, marking lane pixels with white color.
For training I used only those eight images that were provided for testing
the the binary thresholding algorithm. Six of these images were used for
training and two images for validation. For each of these images, I've
created a minified bird eye view image and a corresponding label image. The
minified bird eye view images I put into the [training_data/features]
directory and the label images I put into the [training_data/labels]
directory. Here is an example of the original test image, its bird eye view,
corresponding label image and the output of the trained network:

![Original Test Image][test1_395_222]
![Bird Eye View Image][minified_bird_eye_view_image]
![Bird_Eye_View_Label_Image][bird_eye_view_label_image]
![Network Output][network_output]

All the filtering code: [training] and [predicting] - is consolidated in the
[binary_filter.py] script. I used Keras library on top of TensorFlow. I
tried different optimizers and the Adam optimizer showed the lowest
validation loss. As for the loss function I used the mean squared error. I
was able to achieve the training loss of 0.0015 with the validation loss of
0.0027.

When running the [binary_filter.py] script, it
[generates][save_binary_filter] the [binary_filter.h5] file, that keeps the
trained model, and can be [loaded][load_binary_filter] during the later
stages of the detection pipeline.

[training_data/features]: ./training_data/features
[training_data/labels]: ./training_data/labels
[test1_395_222]: ./examples/test1_395_222.jpg
[minified_bird_eye_view_image]: ./training_data/features/test1.jpg
[bird_eye_view_label_image]: ./training_data/labels/test1.jpg
[network_output]: ./examples/network_output.jpg
[training]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/3e8c47b46bd736a260eb1793ab1664620b179b96/binary_filter.py#L40
[predicting]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/3e8c47b46bd736a260eb1793ab1664620b179b96/binary_filter.py#L94
[binary_filter.py]: ./binary_filter.py
[binary_filter.h5]: ./binary_filter.h5
[save_binary_filter]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/3e8c47b46bd736a260eb1793ab1664620b179b96/binary_filter.py#L73
[load_binary_filter]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/3e8c47b46bd736a260eb1793ab1664620b179b96/binary_filter.py#L91

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I've selected four points on one of the test images with straight lines, and
mapped them to a rectangle on a bird view:

![Test Image][test_image_red_square]
![Bird View][bird_view_red_square]

I hardcoded this points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 560, 475      | 300, 1000     |
| 725, 475      | 670, 1000     |
| 298, 660      | 300, 2220     |
| 1010, 660     | 670, 2220     |

I've calculated the destination coordinated based on the assumption that the
distance between lines is around 3.7 m and the lenght of a single dash line
is around 10 feet, and single pixel of the bird eye view equal to 1 cm. Using
those coordinates, I've [computed][compute_perspective_matrix] the
transformation matrix, that [transforms][create_bird_view] an undistorted
image into a bird eye view image, and the inverse matrix that
[transforms][inverse_bird_view] a bird eye view into the undistorted image.

All code related to the perspective transformation located in the
[bird_view.py] script.

[test_image_red_square]: ./examples/original_red_square.jpg
[bird_view_red_square]: ./examples/bird_view_red_square.jpg
[bird_view.py]: ./bird_view.py
[compute_perspective_matrix]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/45e4e8aa5c15c8b1588b15bdd5c933e3094d80ad/bird_view.py#L26
[create_bird_view]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/45e4e8aa5c15c8b1588b15bdd5c933e3094d80ad/bird_view.py#L29
[inverse_bird_view]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/45e4e8aa5c15c8b1588b15bdd5c933e3094d80ad/bird_view.py#L34

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane-line identifying code resides in the [line_finder.py] script. The
`LineFinder` class takes as the input the thresholded output of the neural
network, calculates histogram of the lower half of it, and
[identifies][identify_line_bases] the possible ares of searching for
the lines, by looking to the two peaks of the histogram. Then the algorithm
performs [sliding window search] in the vertical direction, starting from
the bottom of the image near the previously identified possible area of
the lane line. In each window it [identifies][identify_window_pixels] white
pixels and assumes that these pixels belongs to the line. It calculate the
mean point of all the pixels belonging to the current window and use this
point as the [base of the next window][update_window_base]. As the result,
the `LineFinder` class [returns][return_line_masks] two numpy boolean masks,
each representing pixels of a single lane-line on the thresholded binary
image. Then the algorithm takes a lane-line mask, [identify][non_zero_pixels]
non zero pixels, [scale][scale_mask_pixels] them up, so that we get back to
centimeters, and [fit][fit_line_pixels] the pixels with a polynomial.

[line_finder.py]: ./line_finder.py
[identify_line_bases]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/f108456a82a426b5539e12abeaa6214d682a877c/line_finder.py#L20
[sliding window search]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/f108456a82a426b5539e12abeaa6214d682a877c/line_finder.py#L28
[identify_window_pixels]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/f108456a82a426b5539e12abeaa6214d682a877c/line_finder.py#L33
[update_window_base]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/f108456a82a426b5539e12abeaa6214d682a877c/line_finder.py#L75
[return_line_masks]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/f108456a82a426b5539e12abeaa6214d682a877c/line_finder.py#L18
[non_zero_pixels]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/f108456a82a426b5539e12abeaa6214d682a877c/pipeline.py#L65
[scale_mask_pixels]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/f108456a82a426b5539e12abeaa6214d682a877c/pipeline.py#L66
[fit_line_pixels]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/f108456a82a426b5539e12abeaa6214d682a877c/pipeline.py#L67

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the radius of the curvature I [used][calculate_radius]
the following formula:

![Curvature Radius Formula][curvature_radius_formula]

I [estimate][estimate_radius] the radius by averaging the radius of
the both of the lines, calculating the running average of this value and
round it up to 50 meters.

I [estimate][estimate_position] the position of the vehicle with respect to
the center of the lane by calculating the distance of the lines' middle
point from the middle of the bird view. I also calculate the running average
of this value and round it up to 5 cm.

[calculate_radius]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/ba5fc3a7d1391c84250ae79c5c8d0ca6464c6262/pipeline.py#L128
[curvature_radius_formula]: ./examples/formula.png
[estimate_radius]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/ba5fc3a7d1391c84250ae79c5c8d0ca6464c6262/pipeline.py#L111
[estimate_position]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/ba5fc3a7d1391c84250ae79c5c8d0ca6464c6262/pipeline.py#L119

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the [`Pipeline.process()`][pipeline_process]
function in the [pipeline.py][pipeline_script] script. Here is an example
of my result on a test image:

![Plot Back][plot_back]

[pipeline_process]: https://github.com/mode89/CarND-Advanced-Lane-Lines/blob/ba5fc3a7d1391c84250ae79c5c8d0ca6464c6262/pipeline.py#L21
[pipeline_script]: ./pipeline.py
[plot_back]: ./examples/plot_back.jpg

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
