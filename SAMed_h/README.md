# SAMed_h

## Prerequisites
- Linux (We tested our codes on Ubuntu 18.04)
- Anaconda
- Python 3.10.11
- Pytorch 2.0.0 **(Pytorch 2+ is necessary)**

To get started, first please clone the repo
```
git clone https://github.com/hitachinsk/SAMed.git
cd SAMed_h
```
Then, please run the following commands:
```
conda create -n SAMed_h python=3.10.11
conda activate SAMed_h
pip install -r requirements.txt
```

## Quick start
All the steps are the same as [SAMed](https://github.com/hitachinsk/SAMed). But you need to prepare the [vit_h version of SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints) and [our pretrained checkpoint](https://drive.google.com/file/d/1Kx_vx9bcxJaiMYWAgljNtwtHcooUsq8m/view?usp=sharing).

## Training
We adopt one A100 (80G) for training.
1. Please download the processed [training set](https://drive.google.com/file/d/1zuOQRyfo0QYgjcU_uZs0X3LdCnAC2m3G/view?usp=share_link), whose resolution is `224x224`, and put it in `<Your folder>`. Then, unzip and delete this file. We also prepare the [training set](https://drive.google.com/file/d/1F42WMa80UpH98Pw95oAzYDmxAAO2ApYg/view?usp=share_link) with resolution `512x512` for reference, the `224x224` version of training set is downsampled from the `512x512` version.
2. Run this command to train SAMed.
```bash
python train.py --root_path <Your folder> --output <Your output path> --warmup --AdamW --tf32 --compile --use_amp --lr_exp 7 --max_epochs 400 --stop_epoch 300
```
Check the results in `<Your output path>`, and the training process will consume about 70G GPU memory.

## Difference between SAMed_h and SAMed
- SAMed_h adopt the `vit_h` version of SAM as the base model.
- SAMed_h needs more training iterations. Therefore, we set the max epoch to 400 and early stop to 300 for better performance. 
- Too large learning rate will cause the training instability of SAMed_h. Therefore, we increase the exponent of exponential decay from 0.9 to 7, which can greatly reduce the training instability.
- For faster training speed and less memory consumption, SAMed_h adopts auto mixed-precision, tensor-float 32 and `compile` technology in pytorch 2.0. Therefore, pytorch2+ is necessary for training this model.
