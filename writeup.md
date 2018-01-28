# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./project_submission/images/histogram.png "histogram"
[image2]: ./project_submission/images/rand_img.png "random_images"
[image3]: ./project_submission/images/grayscaled.png "Grayscaled"
[image4]: ./project_submission/images/clahe.png "clahe"
[image5]: ./project_submission/images/clahe_norm.png "clahe_norm"
[image6]: ./test_images/stop.png "stop"
[image7]: ./test_images/double_slope.png "double"
[image8]: ./test_images/priority_road.png "priority"
[image9]: ./test_images/right_of_way.png "right_way"
[image10]: ./test_images/roundabout.png "roundabout"
[image11]: ./test_images/slippery.png "slippery"
[image12]: ./test_images/yield.png "yield"
[image13]: ./test_images/old_yield.png "old_yield"
[image14]: ./test_images/70_limit.png "70_limit"
[image15]: ./project_submission/images/epochvacc.png "epochvsacc"
[image16]: ./project_submission/images/softmax1.png "stop softmax"
[image17]: ./project_submission/images/softmax7.png "double curve softmax"
[iamge18]: ./project_submission/images/softmax9.png "priority softmax"
[image19]: ./project_submission/images/softmax6.png "70km/h softmax"
[image20]: ./project_submission/images/softmax2.png "right of way softmax"
[image21]: ./project_submission/images/softmax4.png "roundabout softmax"
[image22]: ./project_submission/images/softmax5.png "slippery softmax"
[image23]: ./project_submission/images/softmax3.png "yield softmax"
[image24]: ./project_submission/images/softmax0.png "old yield softmax"
[image25]: ./project_submission/images/softmax9.png "priority road softmax"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This writeup should hopefully cover all the rubric points. My code can be found in the file named Traffic_Sign_Classifier.ipynb 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python functions `len()`, `shape()` and `numpy.unique()` to calculate the summary of the data.

* `len()` calculates the length of a data set along given dimension and can be used to find out the number of data points in a given set.
* `shape()` property gives the dimensions of a data set and can be used to calculate the dimensions of an image
* `numpy.unique()` can be used to calculate the unique elements in a data set.

From my calculations, I obtained the following summary of the data set:
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I printed out one random image from each of the data sets to get a sense of what the image looks like. An image showing one such output can be seen below. From the images it can be seen that brightness correction maybe needed to get a lucid image.

![Random images from Training, Validation and Test datasets][image2]

To get a better sense of the input class distribution, I plotted a histogram of the data. From the histogram it can be seen that there is an uneven distribution of the data.

![Histograms of the class distributions][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to first usen the network that was constructed in Lesson 8 with no modifications to the images. I wanted to get a baseline on how bad/good the network would be. The network gave me an accuracy of 47.3% over 50 epochs on the validation data set. (I did not normalize the images either, so that might have contributed to the abysmal accuracy).

After the initial trials with changing weights and increasing/ decreasing the network parameters (Learning rate, number of epochs, number of neurons, etc.), I grayscaled the images and the accuracy imporved from 47.3% to 89.3%. A significant change.

Here is an example of a traffic sign image before and after grayscaling.

![Image before and after grayscaling][image3]

While 89.3% is a significant improvement from before, I decided to further preprocess the data. I felt that a lot of images needed brightness correction so that the lines and curves are better visible. After looking into various methods for brightness correction, I settled on CLAHE correction. A brief explanation and implementation for CLAHE can be found [here](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html).

![Grayscaled image and CLAHE corrected image][image4] 

As a last step, I normalized the image data because I believed that normalizing would make sure that the biases were not too high or too low voiding values obtained for some neurons. For instance, it might be the case that the bias is far greater than the output of the multiplication netween weights and inputs, thereby shadowing them completely. HEre's a CLAHE brightness corrected image and a normalized version of it. 

![CLAHE corrected image and normalized image][image5]

I did not generate any additional data. I will be working on it in the next couple of weeks.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         	|     Description		        	| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 Grayscaled CLAHE Normalized image  	| 
| Convolution     	| 1x1 stride, valid padding, outputs 28x28x16	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 		|
| Convolution	    	| 1x1 stride, valid padding, outputs 10x10x32	|
| Max pooling		| 2x2 stride, outputs 5x5x32			|
| RELU			| 						|
| Flatten and Concatenate | inputs 14x14x32 and 5x5x32, outputs 3936	|
| Fully connected	| inputs 3936, outputs 240	        	|
| RELU			| 						|
| Dropout		| Keep_prob = 0.5				|
| Fully connected 	| inputs 240, outputs 120			|
| RELU			|						|
| Dropout		| Keep_prob = 0.5				|
| Logits output		| inputs 120, outputs 43			|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters:

| Parameter Type		|	Parmeter used					|
|:-----------------------------:|:-----------------------------------------------------:|
| Optimizer			|	Adam optimizer (`tf.train.AdamOptimizer`)	|
| Cost Function			|	`tf.nn.softmax_cross_entropy_with_logits`	|
| Batch size			|	128						|
| epochs			|	50						|
| learning rate			|	0.007						|
| Dropout keep probability	|	0.5						|

I started out with the adam optimizer as it was used in the LeNet lab.  Here's an image capturing the training accuracy vs. epochs:

![epochs vs acc][image15]

After playing around with the network weights and other parameters, I decided to look for other optimizers that might work. I ran the training algorithm various optimizers with cross entropy as the cost function.


The table below shows the validation accuracy for them:

| Optimizer					|	Validation accuracy (%)	|
|:---------------------------------------------:|:-----------------------------:|
|[Gradient Descent Optimizer][1]		|	67.5			|
|[Adadelta Optimizer][2]			|	10.4			|
|[Adagrad Optimizer][3]				|	74.8			|
|[Adam Optimizer][4]				|	**97.8**		|
|[FTRL Optimizer][5]				|	4.8			|
|[Proximal Gradient Descent Optimizer][6]	|	65.4			|
|[Proximal Adagrad Optimizer][7]		|	73.0			|
|[RMS Prop Optimizer][8]			|	**97.2**		|



All results were obtained with the final network architecture discussed above. As it can be seen the Adam Optimizer and RMS Prob Optimizer worked really well compared to the others. However, it is to be noted that I trained the network with no changes to the architecture and hyperparameters. I decided to stick with Adam optimizer as it had better results on the images that I used for testing. 

[1]:https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
[2]:https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer
[3]:https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer
[4]:https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
[5]:https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer
[6]:https://www.tensorflow.org/api_docs/python/tf/train/ProximalGradientDescentOptimizer
[7]:https://www.tensorflow.org/api_docs/python/tf/train/ProximalAdagradOptimizer
[8]:https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **100%**
* validation set accuracy of **97.8%** 
* test set accuracy of **96.3%**

My approach to building the network was iterative with a fair amount of trial and error. The log is as follows:
* Tried the LeNet architecture from Lesson 8 with no modifications. Chosen mostly because it was a good starting point according to the project description.
* The initial architecture had the following results with CLAHE corrected images:
    * Training Accuracy  : **100%**
    * Validation Accuracy : **94.2%**
    * Test Accuracy : **92.2%**

However, it performed very poorly on the new images that I chose with an accuracy of **44.4%**

* I figured the network was overfitting and reduced the number of fully connected layers by 1. The validation accuracy lowered from **94.2%** to **85.6%**. 

* Since lowering the number of layers was not helping, I decided to add another fully connected layer. That did not change the accuracy at all. The accuracy remained unchanged at **92.2%**.

* I increased and decreased the number of neurons, the epochs, batch size, and dropout probabilities. Nothing helped, the accuracy was always around **92%**.

* After several weeks of frustration, I decided to read [Yann LeCunn's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) from the project description to see what he did. His network made sense to me, feeding the outputs of the previous convolution layers to the fully connected layers seemed like a great idea.

* I added the outputs of the previous layers to the network and the accuracy on the testing data increased to about **95%** (I don't remember the exact value). After fine tuning the network, I settled with the results that I stated above. 

* I tuned almost all the parameters in the network, depending on how the network responded. I adjusted the `batch_size` if there was a huge difference in validation accuracy between the epochs. I also varied the learning rate based off the changes in validation accuracy between epochs. I adjusted the number of neurons based on how quickly the network trains vs. the accuracy gains or losses. I adjusted the number of epochs until a seemingly constant slope in validation accuracy was obtained. Unfortunately, I did not keep track of a lot of the changes that I made in the earlier stages. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web and one really old yield sign that I wanted to test the network with:

![Stop sign][image6] ![Double slope][image7] ![Priority road][image8]
![70 speed limit][image14] 
![Right of way][image9] ![Roundabout][image10] ![Slippery road][image11]
![Yield][image12] ![Old yield][image13]

I extracted these images from some photographs found online and I believe none of these are extremely hard to classify. I specifically chose three images with the triangular board (excluding the inverted triangle for the yield sign) to see if the network can differentiate between them. 

1. Image 1 is a stop sign with some changes in orientation and scale. The image has been taken at an angle, thereby changing the overall shape a little. I felt it might be interesting to see how the network responds to such changes without adding any additional augmented data.

2. Image 2 is a double curve image. This image is pretty straight forward. I don't think there's anything out of the ordinary here. The lighting seems adequate too.

3. Image 3 is a priority road sign. The scale on the image is a little streched out and I felt that the right side corner of the sign is buried inside the building behind. I wanted to see if that had any effect on classification.

4. Image 4 is a 70 KM/H speed limit sign. The sign is at a different orientaion and I don't think there's anything out of the ordinary for this one.

5. Image 5 is a right of way road sign. The sign is not at the center of the image. It is slightly towards the right with a scaling difference.

6. Image 6 is a roundabout mandatory sign. It has low lighting compared to the other images that I tested the network with.

7. Image 7 is a slippery road sign. It is also at an angle when compared to the others.

8. Image 8 is a yield sign. It has nothing out of the ordinary.

9. Image 9 is an old style American yield sign. I wanted to test the network on this one to see how it would perform. I wanted to see if the text inside the yield sign made any difference.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        	| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   				| 
| Double curve    		| 	**Children crossing**		|
| Priority road			|	Priority road			|
| 70 km/h	      		| 70 km/h				|
| Right of way			| Right of way      			|
| Roundabout			|	Roundabout			|
| Slippery road			|	Slippery road			|
| Yield				|		Yield			|
| Old yield			|	Yield				|

I highlighted the inaccurate classifications. The model was able to correctly guess 8 of the 9 traffic signs including the old yield, which gives an accuracy of 88.9%. I'm not dissapointed by the accuracy, but I am not thrilled either. As I suspected, it did mix up one of the triangular signs. I think adding more data and also performing some data augmentation might help.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.

For the first image, the model is 99.5% sure that it is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| .9955         	| Stop sign   					| 
| .0019     		| 30 km/h		 			|
| .0013			| 20 km/h					|
| .0008	      		| Roundabout					|
| .00017		| General Caution      				|

Here's an image with the softmax probabilities plotted. Also shown are the input image and an image from the predicted class.

![stop softmax][image16]

For the second image, the model is 99.9% sure that it is a children crossing sign (probability of 0.9999), and the image **does not** contain a priority road sign. The image is of the double curve sign and it does not even show up in the top 5 probabilities. The top five soft max probabilities were

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 0.999        		| Children crossing   				| 
| .00003     		| Right of way		 			|
| .00007		| Beware ice/snow				|
| .00	      		| Road narrows on right				|
| .00			| Dangerous curve to the right			|

Here's an image with the softmax probabilities plotted. Also shown are the input image and an image from the predicted class.

![double curve softmax][image17]

For the third image, the model is 100% sure that it is a priority road (probability of 1.0), and the image does contain a priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        		| Priority road   				| 
| .00     		| Yield			 			|
| .00			| Roundabout					|
| .00	      		| No vehicles					|
| .00			| End of all speed limits			|

Here's an image with the softmax probabilities plotted. Also shown are the input image and an image from the predicted class.

![priority road softmax][image25]

For the fourth image, the model is 99.9% sure that it is a 70km/H speed limit sign (probability of 0.999), and the image does contain a 70km/h speed limit sign. The top five soft max probabilities were

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 0.999        		| 70km/h	   				| 
| .0005     		| 20km/h		 			|
| .0002			| Turn right ahead				|
| .0002	      		| 30km/h					|
| .0000			| keep left					|

Here's an image with the softmax probabilities plotted. Also shown are the input image and an image from the predicted class.

![70km/h softmax][image19]

For the fifth image,  the model is 100% sure that it is a right of way sign (probability of 1.0), and the image does contain a right of way sign. The top five soft max probabilities were

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        		| Right of way   				| 
| .00     		| Pedestrians		 			|
| .00			| Double curve					|
| .00	      		| Beware ice/snow				|
| .00			| Roundabout					|

Here's an image with the softmax probabilities plotted. Also shown are the input image and an image from the predicted class.

![right of way softmax][image20]

For the sixth image, the model is 98.4% sure that it is a roundabout mandatory sign (probability of 0.984), and the image does contain a roundabout mandatory sign. The top five soft max probabilities were

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 0.984        		| Roundabout	   				| 
| .015     		| Priority road		 			|
| .00			| 100km/h					|
| .00	      		| No vehicles					|
| .00			| Vehicles over 3.5 tons prohibited		|

Here's an image with the softmax probabilities plotted. Also shown are the input image and an image from the predicted class.

![roundabout softmax][image21]

For the seventh image, the model is 95.9% sure that it is a slippery road sign (probability of 0.959), and the image does contain a slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 0.959        		| Slippery road   				| 
| .021     		| Double curve		 			|
| .014			| Dangerous curve to left			|
| .003	      		| Beware ice/snow				|
| .0002			| Right of way at intersection			|

Here's an image with the softmax probabilities plotted. Also shown are the input image and an image from the predicted class.

![slippery road softmax][image22]

For the eighth image, the model is 100% sure that it is a yield sign (probability of 1.0), and the image does contain a yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        		| Yield		   				| 
| .00     		| Ahead only		 			|
| .00			| Priority road					|
| .00	      		| Keep right					|
| .00			| Turn left ahead				|

Here's an image with the softmax probabilities plotted. Also shown are the input image and an image from the predicted class.

![yield softmax][image23]

For the ninth image, the model is 99.9% sure that it is a yield sign (probability of 0.999), and the image does contain the old yield sign. Looks like the color and the text inside the sign had no influence. The top five soft max probabilities were

| Probability         	|     Prediction	       			| 
|:---------------------:|:---------------------------------------------:| 
| 0.999        		| Yield		   				| 
| .00005     		| Priority road		 			|
| .00			| Ahead only					|
| .00	      		| No vehicles					|
| .00			| Traffic signals				|

Here's an image with the softmax probabilities plotted. Also shown are the input image and an image from the predicted class.

![old yield softmax][image24]

