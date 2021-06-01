# SPO-dataset
By Or Tslil \\
Sequential Partially Observable dataset for deep temporal images filtering.
This package provide a dataset generator and examples for a sequantial images and their noisy observations. It provides a filtering challenge to filter out nnoisy observations and even partially observable states (see images below). 

## Dataset
Crate a dataset for sequantial images with multiple objects. 
In the images, the state is the images on the bottom (black and white circular objects) and their observations are the top images. Sometimes the state are not observable, for example, in the secong figure the objects pass under the rectangle.

This dataset can benchmark the performance of sequantial estimators (Bayesian filters \ hidden Markov model)

![demo](https://github.com/ortslil64/SPO-dataset/blob/master/images/partal_example_tree.png?raw=true "Under the tree the object are not observable")


![demo](https://github.com/ortslil64/SPO-dataset/blob/master/images/partal_example_checkers.png?raw=true "Partioally observed")


![demo](https://github.com/ortslil64/SPO-dataset/blob/master/images/illusion_example.png?raw=true "Multiple objects with different sizes and different types of observations")

## Usage
A demo dataset can be found in the demo/dataset folder. 
To generate a larger dataset, please refer to demo/dataset_generator.py

## Requairments
To use the dataset generator please install:
- [x] python 3.x
- [x] skvideo
- [x] skimage
- [x] opencv
- [x] imutils
- [x] tqdm
- [x] dlib
- [x] ffmpeg 

## Setup
Clone the repository:
```bash
git clone https://github.com/ortslil64/SPO-dataset
```
Use pip to install:
```bash
pip install -e SPO-dataset
```

