# Data Scientist Project

This is originally a Udacity's Data Scientist Nano-Degree Program project. 

To use this for fun, use command line to run train.py and predict.py. It's better to use the GPU enabled workspace for better performance while training the model.

### Data

The project is to train a classfication model using 102 different types of flowers, where there ~20 images per flower to train on.The dataset can be found [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

The classifier was written in Python, utilizing Numpy, Pytorch and PIL. The pretrained architecture used is `vgg16`, with a default self defined classifier having one two hidden layers with 8192 and 102 nodes respectively. In the command line application, you can specify other existing architectures, and modify the numer of nodes, but not the number of hidden layers (sorry :(). The current accuracy on testset was 87%.

### Copyright

The original design of the project was done by udacity. 

The code was written and tested by myself.

You can find the original repo from [udacity's github](https://github.com/udacity/DSND_Term1).


### Commandline app

1. `train.py`:
  * Basic usage: python train.py data_directory
  * Options:
    * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    * Choose architecture: python train.py data_dir --arch "vgg13"
    * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    * Use GPU for training: python train.py data_dir --gpu
2. `predict.py`
  * Basic usage: python predict.py /path/to/image checkpoint
  * Options:
    * Return top KK most likely classes: python predict.py input checkpoint --top_k 3
    * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    * Use GPU for inference: python predict.py input checkpoint --gpu

