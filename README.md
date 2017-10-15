

# Vehicle Detection Project

[//]: # (Image References)
[image1]: ./output_images/hog_RGB.png
[image2]: ./output_images/hog_HSV.png
[image3]: ./output_images/hog_YCrCb.png
[image4]: ./output_images/hog_LUV.png
[image5]: ./output_images/hog_YCrCb_features.png
[image6]: ./output_images/hog_YCrCb_features.png
[image7]: ./output_images/vehicle_YCrCb_spatial.png
[image8]: ./output_images/non-vehicle_YCrCb_spatial.png
[image9]: ./output_images/vehicle_YCrCb_hist.png
[image10]: ./output_images/vehicle_raw_scaled.png
[image11]: ./output_images/sliding_window_overlapped.png
[image12]: ./output_images/test_detection.png
[image13]: ./output_images/vehicle_heat_boundary.png
[image14]: ./output_images/vehicle_img.png
[image15]: ./output_images/non_vehcile_img.png
[video1]: ./output_videos/project_video.mp4

---
### Writeup / README

This write up is is also available [here].(https://github.com/geek101/CarND-Vehicle-Detection/blob/master/README.md)  

You're reading it!

### Histogram of Oriented Gradients (HOG)

The code for this step is contained in the code cell 6 of the IPython notebook [here](https://github.com/geek101/CarND-Vehicle-Detection/blob/master/vehicle_detection.ipynb).

I started by loading all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image14]

![alt text][image15]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]

#### HOG parameters.

The best parameters seem to be `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and for color `YCrCb` where there some variability in the hog features but still there some similarity between hog features of each channel.

Increasing the pixes per cell did not help much and reducing orientations has made the image more sparse.

HOG features vector of all channels plotted:

![alt text][image5]

#### Color histogram for various color spaces

YCrCb color histogram is also added to the feature vector. Its histogram
plotted for a non vehicle:

![alt text][image9]

#### Spatial Binning of Color

The following image shows the plots for vehicle and non vehicle color bins. Images are resized to 32x32

![alt text][image7]

![alt text][image8]

#### Feature normalization.

Since we are using 3 different features StandardScaler() is used to scale feature vectors. The following shows the raw and normalized feature vector plot.

![alt text][image10]

#### SVM model training.

GridSearch/RandomSearch with the following features(refer code cell 27):

`{'C': [1, 10, 100], 'gamma': [1.0, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear']}`
yielded the best params:
`{'C': [10], 'gamma': [0.0001], 'kernel': ['rbf']}` and test error was at 99.8% however the model performance we quite poor.

Hence I have used a linear SVM model instead and it yielded test error 99

### Sliding Window Search

I have used the efficient sliding window implementation as described in the lesson where hog features are first extracted for the entire image and window is then run across them. This is done for all three channels.

The following are the various scale parameters that I have used the window search to help with vehicle detection. This seems to work best without having too many false positives and enough overlap of vehicle detection so that thresholding helps remove false positives.

`[0.75, 1.0, 1.5, 2.0, 2.5]`

![alt text][image11]

#### Few more examples of vehicle detection using sliding window.

Once can notice false positives.
![alt text][image12]

---

### Video Implementation

Here's a [link to my video result](./output_videos/project_video.mp4)
Threshold of 4 seems to provide best result with almost no false positives.

#### Heatmap and thresholding for better results.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the image:

![alt text][image13]

---

### Discussion

Trying out various combinations of color spaces was quite challenging. I still quite not sure why YCrCb works quite well when compared to HSV for example. Perhaps it might work better if I used different color spaces for each of the three features.

Quite surprised at the effectiveness of SVM specially after working with CNNs in the last couple of projects.

Next step in the project is to train a CNN for which more image samples are required specially for non-vehicle since both the extended datasets provided only contain images for vehicles. Perhaps SVM can help collect non-vehicle samples from those images.

 

