# Data Scientist Project

This is original a Udacity's Data Scientist Nano-Degree Program project. 

To use this for fun, use command line to run train.py and predict.py. It's better to use the GPU enabled workspace for better performance while training the model.

### Data

The project is to train a classfication model using 102 different types of flowers, where there ~20 images per flower to train on.The dataset can be found [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

The classifier was written in Python, utilizing Numpy, Pytorch and PIL. The pretrained architecture used is `vgg16`, with a default self defined classifier having one two hidden layers with 8192 and 102 nodes respectively. In the command line application, you can specify other existing architectures, and modify the numer of nodes, but not the number of hidden layers (sorry :(). The current accuracy on testset was 87%.

### Copyright

The original design of the project was done by udacity. 

The code was written and tested by myself.

You can find the original repo from [udacity's github](https://github.com/udacity/DSND_Term1).
