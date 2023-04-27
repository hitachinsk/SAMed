# Data preprocessing
This README file records the operation of data preprocessing, corresponding to the operations in [preprocess_data.py](preprocess_data.py)
1. Please download the Synapse dataset from the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/). Convert them to numpy format, clip the value range to \[-125,275\], normalize the 3D image to \[0-1\], extract the 2D slices from the 3D image for training and store the 3D images as h5 files for inference.
2. According to the [data description](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) of the Synapse dataset, the map of the semantic labels between the original data and the processed data is shown below.

Organ | Label of the original data | Label of the processed data
------------ | -------------|----
spleen | 1 | 1
right kidney | 2 | 2
left kidney | 3 | 3
gallbladder | 4 | 4
liver | 6 | 5
stomach | 7 | 6
aorta | 8 | 7
pancreas | 11 | 8
