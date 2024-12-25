## Introduction
This is the implementation of the SGD-![](http://latex.codecogs.com/svg.latex?r\\alpha) algorithm.

## Installation
We assume that you're using [Python 3.6+](https://www.python.org/downloads/) with [pip](https://pip.pypa.io/en/stable/installing/) installed. You need to run the following inside the root directory to install the dependencies:

```bash
pip install torch, numpy, scipy, scikit-learn
```
This will install the following dependencies:
* [torch](https://github.com/fchollet/keras) (the library was tested on version 1.7.0 but anything above 1.5.0 should work)
* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)


## Testing
We run the experiments on [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/), and to run the test you should download datasets to the [Datasets/](Datasets) directory(there is an example dataset [ijcnn1](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1) in it). After downloading, change the dataset name --`dsetName`-- and the features size --`featureSize`-- in the [main.py](main.py) (do not change anything if you choose to use the example dataset)and run:

```bash
python main.py
```







