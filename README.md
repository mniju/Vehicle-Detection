## Vehicle Detection Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the above pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./output_images/car_nocar.png
[image2]: ./output_images/Spatiallybinned.png
[image3]: ./output_images/ColorHistogramYrcb.png
[image4]: ./output_images/Hog1.png
[image5]: ./output_images/normalized.png
[image6]: ./output_images/heatmap.png
[image7]: ./output_images/final.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


###Histogram of Oriented Gradients (HOG)

####1. HOG features from the training images.


The code for this step is contained in  *cell9* of the IPython notebook  `Vehicle Detection.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images. Ifound the total no of cars and nocars images, their datatype and image size.This is calculated in *cell3* of the IPython notebook  `Vehicle Detection.ipynb`.  


	No of Cars: 8792
	No of NonCars: 8968
	Image Shape: (64, 64, 3)
	Image Type: float32

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

At first, i did **Spatial Binning** and **Color Histogram**
##### Spatial Binning:
I resized the image to `(32x32) from (64x64)` and then added all the three channels one after the other in an array(*cell7* of the IPython notebook  `Vehicle Detection.ipynb`.).Here is the Spatially binned data for a car in different channels.As we can see, RGB doesnt give good feature. HSV or YCrCb gives good features.

![alt text][image2]

##### Color Histogram:
Next i tried to collect a Histogram of Colors.Since we have color image, there is a possiblity we can find the vahicle with features extracted based on colors.(*cell8* of the IPython notebook  `Vehicle Detection.ipynb`.)The Histogram of three channel for a random image in `YCrCb` colorspace is shown below.
![alt text][image3]

##### HOG:
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.(*cell9* of the IPython notebook  `Vehicle Detection.ipynb`.)

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image4]

#### 2. final choice of HOG parameters.

I tried various combinations of parameters . RGB Color space was working only for bright images. For Blurry and dark images, the Hog features were not satisfactory in RGB. I tried with HSV and YrCrb and settled with `YrCrb` color space.
Finally i ended up  feature extraction with the following settings

	colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 9
	pix_per_cell = 8
	cell_per_block = 2
	hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
	size=(32, 32) # 16,16
	nbins = 32 # 16
	n_samples = 4000



#### 3.  Train a classifier using your selected HOG features (and color features if you used them).

I used a combination of Spatial Binnng, Color Histogram and HOG with all the channels.(*cell10* of the IPython notebook  `Vehicle Detection.ipynb`.) for feature extraction

I extract each feature(spatial_features, colourhistogram,hog_features) for an image and combine them  in line `features.append(np.concatenate((spatial_features, hist_features,hog_features)))` (*cell27* of the IPython notebook  `Vehicle Detection.ipynb`.)

After extracting the features, i normalized the features using `StandardScaler` in `sklearn.preprocessing` 
(*cell28* of the IPython notebook  `Vehicle Detection.ipynb`.)

	X = np.vstack((car_features, notcar_features)).astype(np.float64) 	
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)

Here is a sample image of Normalized and Raw feature for a CarImage

![alt text][image5]

I splitted 20% of the training data as test data.
`X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)`
I used Linear SVC from `sklearn` package and trained the data using `svc.fit(X_train, y_train)`.(*cell29* of the IPython notebook  `Vehicle Detection.ipynb`.)When i tried to train all the data it took a long time. I played with subset of data to train wherin it will detect the vehicles in the end properly.I started with 100.then 1000. Vehicle detection happend properly for 4000 images of car and non cars. So i stayed with it 4000 data.*cell27*

	n_samples = 4000
	random_idxs = np.random.randint(0,len(cars),n_samples)
	test_cars = np.array(cars)[random_idxs]
	test_notcars = np.array(notcars)[random_idxs]

I got **99.25 %** accuracy.Training output shown below.

	Fitting in progress...
	19.81 Seconds to train SVC...
	Test Accuracy of SVC =  0.9925

### Sliding Window Search

#### 1. Sliding window search.  what scales to search and how much to overlap windows?

I used Hog Subsampling Window as it was mentioned to be faster and effective.We do a HOG feature extraction for a portion of the image and subsample that further to match the features. This is quite effective rather than doing HOG segme.nt by segment in the test image.The Subsamlping feature is implement in the function
 `def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):`.

(*cell30* of the IPython notebook  `Vehicle Detection.ipynb`.)This function takes many  inputs.

* `img` - Input image (from video) where the vehicle is to be searched.
* `ystart` - Yaxis (Top) of the image where the search to be started.`ystart = 400`
* `ystop` - Yaxis (Botom) of the image where the search to be stopped.`ystop = 656`
* `scale` - Scaling to be done on the image search area before starting the search.`ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))`.
* `svc` - Trained Weights.
* `X_scaler` - Scaling value.
* `orient`, `pix_per_cell`, `cell_per_block` - HOG Extraction parameters.
* `spatial_size` - Spatial Binning feature extraction `(32,32)`
* `hist_bins` - Parameter for Color Histogram feature extraction

##### Inside the `find_cars` Function:
Code in *cell39* of the IPython notebook  `Vehicle Detection.ipynb`.

First i normalize the image.`img = img.astype(np.float32)/255`

After this i extract the slice of the image to be searched `img_tosearch = img[ystart:ystop,:,:]`

Next perform a colorspace conversion if required. I settled with  YrCrb.`ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')`.

Apply scaling before doing the search `ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))`

Now extract the three channels from the image seperately.

	ch1 = ctrans_tosearch[:,:,0]
	ch2 = ctrans_tosearch[:,:,1]
	ch3 = ctrans_tosearch[:,:,2]

Find the x and y blocks and x and y steps from `pix_per_cell` and `cells_per_step` respectively.
	nxblocks = (ch1.shape[1] // pix_per_cell)-1
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step

After this i do a HOG feature extraction  for the complete image.` hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)` 

After this i go each steps in X and Y position and extract all the features.

    for xb in range(nxsteps):
        for yb in range(nysteps):
        	#subsample Hog
    		#Extract spatial and Color Histogram Features
    		#Combine all the features
    		#Scale all the features using Xscaler
    		test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
    		#Do Prediction
    		test_prediction = svc.predict(test_features)

when we get a prediction

	  if test_prediction == 1:
		#Calculate border, draw Rectangle
	    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
	    img_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
		#Increment 1 in the Heat map array.
	    heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] +=1

Heat map is used as a means to reduce False positives.When required we can have a threshold remove lesser confident boxes by thresholding the Heat Map.See *cell40*

When i used a scale of 1 , bigger cars are not detected . When i used a larger value , smaller cars are not getting detected. I found somwhere i should use around 1.4 ~ 1.6 scaling. I settled with 1.5 scale for the test images to be detected properly.Here is the sliding boxed detection and Heat map for all the example images.


![alt text][image6]

There false detection in one image. Overall , this looks good.

#### 2. Some examples of test images to demonstrate how  pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. In addition to above pipeline, i used the `labels` from `scipy.ndimage.measurements` to get the labels from the heatmap and then draw a single box for a vehicle detection.

 Here are some example images:

![alt text][image7]

The output seems to be decent enough to detect vehicles in road.
---

### Video Implementation

#### 1. Link to your final video output. (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
I put together the pipeline to run on the project video.(*cell59* of the IPython notebook  `Vehicle Detection.ipynb`.)At miss the car detection an second or two in between . My pipeline does pretty well with the project video.

Here's a [link to my video result](./project_video.mp4)


#### 2. some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  (*cell42* of the IPython notebook  `Vehicle Detection.ipynb`.)

Here's an example result showing the heatmap from a series of example images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid 

### Here are the six test images and their corresponding heatmaps:

![alt text][image6]


### Here the resulting bounding boxes are drawn to each test image:
![alt text][image7]

---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As seen above, my pipeline with the defined parameters works as shown .At times it misses to cath a car when it is in the half sideways. This might be because, I used only 4000 images to train.When i use the complete test data, there wont be any misses.At that time i may have to implement more robust false detection.

