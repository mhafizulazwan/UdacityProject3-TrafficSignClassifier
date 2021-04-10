# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/training_image.png "Training"
[image2]: ./examples/frequency_data.png "Bar Chart"
[image3]: ./new_image/new_image1.png "Speed limit"
[image4]: ./new_image/new_image2.png "Turn right"
[image5]: ./new_image/new_image3.png "Slippery road"
[image6]: ./new_image/new_image4.png "No entry"
[image7]: ./new_image/new_image5.png "Traffic light"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python method to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The following is a random image from the training set:

![alt text][image1]

Below is a bar chart showing the frequency of each class.

![alt text][image2]

As the frequency of the data is not balance, future work should find more images or use some image augmentation techniques to balance the frequency of the data. By doing this, it might improve the performance of the model.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it can reduce the value of each pixel in reach image into 8-bits instead of 24-bits. As a result, this might reduce the processing speed as well.

As a last step, I normalized the image data because it can reduce the large values in the inputs.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers (5 layers):

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3      	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| simplest non-linear activation function		|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16  	|
| RELU					| simplest non-linear activation function		|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten       		| outputs 400  									|
| Fully connected		| outputs 120        							|
| RELU					| simplest non-linear activation function		|
| Fully connected		| outputs 84        							|
| RELU					| simplest non-linear activation function		|
| Fully connected		| outputs 43        							|
| Softmax				| softmax with cross entropy					|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer, the batch size of 128, 30 epochs, and 0.001 learning rate.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 93.6%
* test set accuracy of 91.1%

* The chosen architecture was LeNet from Yann Lecun
* I believe it would be relevant to the traffic sign application because it is a non-linear model and thus, it is capable to solve much more complex problems. 
* Because the final model's accuracy on the training, validation and test set is above 90, I think the model is working well.
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

The first image might be difficult to classify because the training data is little (see the frequency of data previously).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| No entry   									| 
| Traffic light 		| Traffic light 								|
| Speed limit (20 km/h)	| Speed limit (20 km/h)							|
| Turn right ahead		| Turn right ahead				 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess all of the traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 91.1%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model is certain that this is a no entry sign (probability of 0.99), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| No entry   									| 
| .0008     			| Stop sign 									|
| .00006				| Slippery road									|
| .000007	      		| End of no passing				 				|
| .0000007				| Roundabout mandatory 							|

For the second image, the model is certain that this is a traffic light sign (probability of 0.99), and the image does contain a traffic light sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Traffic light 								| 
| .00001     			| General caution 								|
| .0000000009			| Pedestrians									|
| .0000000003	      	| Road narrows on the right		 				|
| .000000000001			| Go straight or left 							|

Reader can see the output of 14th cell in the Ipython notebook for the softmax probabilities of the third until fifth images. You can see that the model is quite certain about its prediction. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


