# **Traffic Sign Recognition**

### Goals which were acquired and achieved in this project:

- Load the data set (see below for links to the project data set)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

[//]: # "Image References"
[imagevisualization]: ./readme-images/dataset-histogram.png "Visualization"
[imagedataset]: ./readme-images/dataset-examples.png "DatasetExamples"
[imagepipeline]: ./readme-images/image-after-pipeline.png "ImagesAfterPipeline"
[imagetrafficsigns]: ./readme-images/5-germain-traffic-signs.png "GermanTrafficSigns"
[imagetpredicitons]: ./readme-images/signs-precitions.png "SignsPredictions"

---

## Rubric Points

### Submitted Files

1.  [README.md](https://github.com/alexstelya/CarND-Traffic-Sign-Classifier-Project/blob/master/README.md) which explain in details how traffic sign recognition model was created, trained and tested
1.  [Jupyter notebook](https://github.com/alexstelya/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
1.  [HTML output of the code](https://github.com/alexstelya/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Dataset Summary

I used the numpy library to calculate summary statistics of the traffic
signs data set:

- The size of training set is 34799
- The size of the validation set is 4410
- The size of test set is 12630
- The shape of a traffic sign image is (32, 32, 3)
- The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing how the data distributed across different traffic sings.

![alt text][imagevisualization]

Here are a few examples how signs are actually looks like in dataset.

![alt text][imagedataset]

### Design and Test a Model Architecture

#### 1. Preprocessing

I decided to go with simple approach of grayscaling and normalizing dataset images.
I have created a simple function image_pipeline(image_data) which will be used for all images provided for my model.
I used this approach because we need normalized data to have zero mean and equal variance.
As for grayscale, I see this as a common practice to speed up calculations and remove unwanted details.

Here is an example of the processed image.

![alt text][imagepipeline]

#### 2. Model Architecture

The model architecture is based on the LeNet model architecture. I decided to go with tensorflow 2.0 and created my model in sequential mode. I added dropout layers before each fully connected layer to improve my model performance.

Here is summary of my model:

---

| Layer (type)        | Output Shape       | Param # |
| ------------------- | ------------------ | ------- |
| conv2d              | (None, 28, 28, 6)  | 456     |
| average_pooling2d   | (None, 14, 14, 6)  | 0       |
| conv2d_1            | (None, 10, 10, 16) | 2416    |
| average_pooling2d_1 | (None, 5, 5, 16)   | 0       |
| flatten             | (None, 400)        | 0       |
| dense               | (None, 120)        | 48120   |
| dropout             | (None, 120)        | 0       |
| dense_1             | (None, 84)         | 10164   |
| dropout_1           | (None, 84)         | 0       |
| dense_2             | (None, 43)         | 3655    |

---

Total params: 64,811  
Trainable params: 64,811  
Non-trainable params: 0

---

#### 3. Model training

To train the model, I used an Adam optimizer and Categorical Cross Entropy for loss calculation. After some tweaking I leave the hyperparameters as follows:

- batch size: 128
- number of epochs: 80
- learning rate: 0.0006
- mu = 0.0
- sigma = 0.1
- dropout = 0.5

To train the model, I used an iterative approach. First I started with well-known LeNet architecture. On start accuracy was not impressive - about 85%. I decided to drop learning rate to 0.0008. Then I started tweeking number of epochs. However, I couldn't get past the 90% barrier. Because of that, I decided to add 2 dropout layers and reduced the learning rate to 0.0006. After all these changes, my final model results were:

- training set accuracy of 98.85%
- validation set accuracy of 96.19%
- test set accuracy of 93.39%

### Test a Model on New Images

#### 1. Here are five German traffic signs that I found on the web:

![alt text][imagetrafficsigns]

I decided to go with some common traffic signs. I think the most problematic signs will be 4-th one becuase it is very similar to other signs. Also on this particular picture we can see that sign has yellow background which may be confusing for classifier.
Other sign should not make any problem to the model, however "Stop" sign can me misclassified for "No entry" sing, and a "Turn left" sign can be classified as sign with some other direction.

#### 2. Here are the results of the prediction(probability added under each image):

![alt text][imagetpredicitons]

The model correctly classified 4 out of 5 traffic signs, resulting in an accuracy of 80%. Although this is lower than the accuracy on the test set (which was 93.39%), I believe that the model will perform better on a larger set of new images. As expected, the model misclassified the 'Beware of ice' sign, predicting with 51% probability that it was the 'Bicycles crossing' sign.

#### 3. The code for making predictions on my final model is located in the 36th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 99.89%), and the image does contain a stop sign. The top five soft max probabilities were:

| Probability |      Prediction      |
| :---------: | :------------------: |
|  99.89522%  |      Stop sign       |
|  0.10410%   |      Keep right      |
|  0.00036%   | Speed limit (30km/h) |
|  0.00016%   |     No vehicles      |
|  0.00013%   |    Priority road     |

For the second image (Yield) model made a perfect prediction with 100% probability:

| Probability |    Prediction    |
| :---------: | :--------------: |
| 100.00000%  |      Yield       |
|  0.00000%   |    Ahead only    |
|  0.00000%   |    Keep right    |
|  0.00000%   |   No vehicles    |
|  0.00000%   | Turn right ahead |

Same was for the third sign (Turn left ahead):

| Probability |    Prediction     |
| :---------: | :---------------: |
| 100.00000%  |  Turn left ahead  |
|  0.00000%   |    Keep right     |
|  0.00000%   | End of no passing |
|  0.00000%   |    No passing     |
|  0.00000%   |    Ahead only     |

Model did fail on forth image(Beware of ice/snow).
Correct answer was only on 3-rd place:

| Probability |          Prediction          |
| :---------: | :--------------------------: |
|  51.30662%  |      Bicycles crossing       |
|  48.62304%  |        Slippery road         |
|  0.05270%   |      Beware of ice/snow      |
|  0.01316%   |          Road work           |
|  0.00276%   | Dangerous curve to the right |

The fifth image (Priority road.png) was correctly predicted with 99.99% certainty:

| Probability |             Prediction              |
| :---------: | :---------------------------------: |
|  99.99996%  |            Priority road            |
|  0.00004%   |        Roundabout mandatory         |
|  0.00000%   |             No vehicles             |
|  0.00000%   |        Speed limit (100km/h)        |
|  0.00000%   | End of all speed and passing limits |

## Conclusion

Through this project, I was able to create and train a traffic sign classification model with a test set accuracy of 93.39%. The model successfully classified 4 out of 5 randomly-selected traffic signs found on the internet. While I consider this a success, there is still room for improvement. One obvious improvement would be to add a few more layers and increase the number of epochs to see if the model's performance can be further improved.
