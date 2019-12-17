# FPN
```
Pytorch Implementation of "Feature Pyramid Networks for Object Detection"
You can star this repository to keep track of the project if it's helpful for you, thank you for your support.
```


# Environment
```
OS: Ubuntu 16.04
Python: python3.x with torch==0.4.1, torchvision==0.2.2
```


# Performance
|  Backbone      | Train       |  Test         |  Pretrained Model  |  Epochs  |	Learning Rate		|    RoI per image   |   AP      					|
|  :----:        | :----:      |  :----:       |  :----:    	    |  :----:  |	:----:				|   :----:  		 |   :----: 				    |
| ResNet50-FPN   | trainval35k |  minival5k    |  Pytorch		    |  12	   |	2e-2/2e-3/2e-4   	|	128              |   -                          |
| ResNet101-FPN  | trainval35k |  minival5k    |  Pytorch   	    |  12	   |	2e-2/2e-3/2e-4		|	128  			 |	 -							|


# Trained models
```
You could get the trained models reported above at 
```


# Usage
#### Setup
```
cd libs
sh make.sh
```
#### Train
```
usage: train.py [-h] --datasetname DATASETNAME --backbonename BACKBONENAME
                [--checkpointspath CHECKPOINTSPATH]
optional arguments:
  -h, --help            show this help message and exit
  --datasetname DATASETNAME
                        dataset for training.
  --backbonename BACKBONENAME
                        backbone network for training.
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to use.
```
#### Test
```
usage: test.py [-h] --datasetname DATASETNAME [--annfilepath ANNFILEPATH]
               [--datasettype DATASETTYPE] --backbonename BACKBONENAME
               --checkpointspath CHECKPOINTSPATH [--nmsthresh NMSTHRESH]
optional arguments:
  -h, --help            show this help message and exit
  --datasetname DATASETNAME
                        dataset for testing.
  --annfilepath ANNFILEPATH
                        used to specify annfilepath.
  --datasettype DATASETTYPE
                        used to specify datasettype.
  --backbonename BACKBONENAME
                        backbone network for testing.
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to use.
  --nmsthresh NMSTHRESH
                        thresh used in nms.
```
#### Demo
```
usage: demo.py [-h] --imagepath IMAGEPATH --backbonename BACKBONENAME
               --datasetname DATASETNAME --checkpointspath CHECKPOINTSPATH
               [--nmsthresh NMSTHRESH] [--confthresh CONFTHRESH]
optional arguments:
  -h, --help            show this help message and exit
  --imagepath IMAGEPATH
                        image you want to detect.
  --backbonename BACKBONENAME
                        backbone network for demo.
  --datasetname DATASETNAME
                        dataset used to train.
  --checkpointspath CHECKPOINTSPATH
                        checkpoints you want to use.
  --nmsthresh NMSTHRESH
                        thresh used in nms.
  --confthresh CONFTHRESH
                        thresh used in showing bounding box.
```