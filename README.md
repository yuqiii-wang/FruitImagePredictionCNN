# FruitImagePredictionCNN
This is an attempted solution to classification of 18 fruit categories via bilinear CNN, with an built app able to make real time prediction. The data set consists of 2400 fruit images from various sources. The top prediction accuracy is 93%.

#### About main_code.py
This .py file lists all developed CNN models for the Fruit prediction. However, there are improvements: larger batch_size, more crop solutions that generates/affines to more images.

#### About the prediction app
This is a simple app built on Flask framework and uses bootstrap4 for front end display. It can take images from user's webcam and make prediction on demand. A demonsration of this app can be seen here https://youtu.be/6i5DglbV2lg .