## Traffic Sign Recognition 

The goal of this project is to develop a deep neural network classifier for the German Traffic Sign Recognition Benchmark (GTSRB). The well known LeNet-5 network architecture will be used as a basis for this task.

Some of the goals of this project are to:
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

The whole project is contained in [this jupyter notebook](https://github.com/Corni33/CarND_P2_TrafficSigns/blob/master/Traffic_Sign_Classifier.ipynb).
In addition to code, the notebook also contains explanations and rationales.
In the following writeup some additional commentary is given.

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---

## Writeup 

### Data Set Summary & Exploration

The GTSRB data set contains 34799 training examples, 4410 validation examples and 12630 testing images that each belong to one of 43 classes. 
All images in the data set have a size of 32x32 pixels. 
These statistics are calculated in the first code cell of the notebook.

An exploratory visualization of the GTSRB data set is performed in the code cells 2 to 5 of the notebook.
After examining the number of training examples per class, some of the classes with relatively many and some of the classes with relatively few examples are sampled and plotted. 


### Design and Test a Model Architecture

#### **1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.**

The preprocessing steps, which include grayscaling and normalization of the image pixel data, are contained in the code cells 6 to 8 of the IPython notebook.


#### **2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)**

During examination of the training data it became apparent, that some classes have a lot less training examples than others. 
While training the network, these rare classes would therefore occur less often, have less influence on the loss function and as a result would probably be predicted with less accuracy after training has finished. 
In order to improve classification for these rare classes, their number of training examples has been artificially increased by duplicating existing examples. 
The code for this operation is contained in code cell 9 of the IPython notebook.  


#### **3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.**

The employed neural network model architecture is based on the LeNet-5 architecture.
After adjusting merely the number of output classes, the model showed a validation accuracy of about 90%. 

To further enhance image classification the convolutional layers have been made deeper and an additional fully connected layer has been added. 
As these changes introduced many new degrees of freedom, the model started to overfit which caused the validation accuracy to decline.
To prevent the model from overfitting, dropout has been added to all fully connected layers. 
The model now reached a validation accuracy between 96 and 97%.

The code for the final model is located in cell 10 of the notebook. The model consits of the following layers:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 grayscale image   							                 | 
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					       |												                                     |
| Max pooling	2x2 | 2x2 stride, valid padding, outputs 14x14x10 				 |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x20 	|
| RELU					       |												                                     |
| Max pooling	2x2 | 2x2 stride, valid padding, outputs 5x5x20 				   |
| Fully connected		| input = 500, output = 120        					|
| RELU					       |												                                  |
| Dropout					       |												                               |
| Fully connected		| input = 120, output = 84        					|
| RELU					       |												                                  |
| Dropout					       |												                               |
| Fully connected		| input = 84, output = 60        					|
| RELU					       |												                                  |
| Dropout					       |												                               |
| Fully connected		| input = 60, output = 43        					|
| Softmax				     |         									|
 


####**4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.**

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####**1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.**

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####**2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).**

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####**3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)**

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
