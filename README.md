# Test-Linking-Segments-Text-Localization

This repo serves as a record of the prcess that I took to test the Linking-Segments model illustrated in this paper (https://arxiv.org/abs/1703.06520) and implemented in this repo (https://github.com/bgshih/seglink)

## General Results
Model is trained using ICDAR_2015 dataset. With one NVIDIA Tesla P-100 GPU, the training process took about 10 hours using the default hyperperameter settings specified at (https://github.com/bgshih/seglink/blob/master/exp/sgd/finetune_ic15.json). Loss change during the training can be seen down below. 
![alt text](https://github.com/Yiming992/Test-Linking-Segments-Text-Localization/blob/master/loss_curve.png)
Moreover, **results_20180605_081219.zip** conatins evaluation results of the trained model on all test images of ICDAR_2015 dataset, produced by evaluation script located at(https://github.com/bgshih/seglink/blob/master/seglink/evaluate.py)


## Steps And Issues Encountered

### Step I: Set up environment

Training environment was set up in a Gcloud Compute Engine with following sepcs:

+ Ubuntu 16.04
+ Nvidia Tesla P-100, 16 GB memory
+ 64 GB persistent storage
+ 8 CPU cores with 16 GB memory
+ CUDA 8.0
+ CUDNN 6.0
+ Python 3.4
+ Tensorflow 1.3

Although the model was only tested on Ubuntu 14.04, I have seen no visiable issues when compile and train on Ubuntu 16.04.

A few issues I encountered during the setup. Firstly, there seems to have a bug that prevents Nvidia driver to communicate with P-100 GPU on ubuntu 14.04, so I decided to go with ubuntu 16.04 instead. Secondly, although in the seglink repo, author states the model is for tensorflow version 1.0 or newer, going above 1.3 will result in a error related to the custom c++ op, and consequently, CUDA 9.0 will not be supported.
### Step II: Create Datasets
This step is relatively straightforward. Just Download required datasets, and then run this script https://github.com/bgshih/seglink/blob/master/tool/create_datasets.py, with your own file path and datasets, to transform data into TFRecords for tensorflow

### Step III: Download model checkpoints

Firstly, There are not model checkpoints in the original repo, so everything must be downloaded  There are two routes to take to train the model

#### Route I:

+ Download VggNet from (https://github.com/conner99/VGGNet)
+ Convert caffemodel to tensorflow using files in (https://github.com/bgshih/seglink/tree/master/tool/convert_caffe_model)
+ Pretrain on synthtext dataset using settings in (https://github.com/bgshih/seglink/blob/master/exp/sgd/pretrain.json)
+ Finetune on ICADR_15 data using settings in (https://github.com/bgshih/seglink/blob/master/exp/sgd/finetune_ic15.json)


#### Route II:

+ Download VGG16 pretrained on synthtext directly from (https://pan.baidu.com/s/1nvuWqlr)
+ Finetune on ICADR_15 data using settings in (https://github.com/bgshih/seglink/blob/master/exp/sgd/finetune_ic15.json)

Personally I went with Route II to save time computational resources for now

### Step IV: Train and evaluate the model

To train the model, firstly make sure the file paths in (https://github.com/bgshih/seglink/blob/master/exp/sgd/finetune_ic15.json) and https://github.com/bgshih/seglink/blob/master/seglink/solver.py are correct. Then run
`./manage.py train exp/sgd finetune_ic15`.

To Evaluate the model, simply run `./manage.py test exp/sgd test_ic15`. However, during my evaluation, I have to resize images to 512*512 to avoid an out of memory error


## Future Plan
I am planning to carry out following two tasks in the near future

### Task I:
Tune hyperparameters in (https://github.com/bgshih/seglink/blob/master/exp/sgd/finetune_ic15.json), to improve finetune performance

### Task II:

Train the complete pipeline from pretrain, finetune to evaluation

## Additional advice

During my exploration, I also found several useful discussions between other users. These discussions can be found at following links
+ (https://github.com/bgshih/seglink/issues/3)
+ (https://github.com/bgshih/seglink/issues/1)
+ (https://github.com/bgshih/seglink/issues/4)


















