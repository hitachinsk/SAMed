# SAMed
This repository contains the implementation of the following paper:
> **Customized Segment Anything Model for Medical Image Segmentation**<br>
> [Kaidong Zhang](https://hitachinsk.github.io/), and [Dong Liu](https://faculty.ustc.edu.cn/dongeliu/)<br>
> Technical report<br>
<img src="materials/teaser.png" height="140px"/> 

## Overview
<img src="materials/pipeline.png" height="260px"/> 
We propose SAMed, a general solution for medical image segmentation. Different from the previous methods, SAMed is built upon the large-scale image segmentation model, Segment Anything Model (SAM), to explore the new research paradigm of customizing large-scale models for medical image segmentation. SAMed applies the low-rank-based (LoRA) finetuning strategy to the SAM image encoder and finetunes it together with the prompt encoder and the mask decoder on labeled medical image segmentation datasets. We also observe the warmup finetuning strategy and the AdamW optimizer lead SAMed to successful convergence and lower loss. Different from SAM, SAMed could perform semantic segmentation on medical images. Our trained SAMed model achieves 81.88 DSC and 20.64 HD on the Synapse multi-organ segmentation dataset, which is on par with the state-of-the-art methods. We conduct extensive experiments to validate the effectiveness of our design. Since SAMed only updates a small fraction of the SAM parameters, its deployment cost and storage cost are quite marginal in practical usage.

## Todo list
- [ ] Make a demo.
- [ ] Finetune on more datasets
- [ ] Make SAMed based on `vit_l` or `vit_h` mode of SAM

## Prerequisites
- Linux (We tested our codes on Ubuntu 18.04)
- Anaconda
- Python 3.7.11
- Pytorch 1.9.1
To get started, first please clone the repo
```
git clone https://github.com/hitachinsk/SAMed.git
```
Then, please run the following commands:
```
conda create -n SAMed python=3.7.11
conda activate SAMed
pip install -r requirements.txt
```

If you have the raw Synapse dataset, we provide the [preprocess script](preprocess/) to process and normalize the data for training. Please refer this folder for more details.

## Quick start
1. Change the directory to the rootdir of this repository.
2. Please download the pretrained [SAM model](https://drive.google.com/file/d/1_oCdoEEu3mNhRfFxeWyRerOKt8OEUvcg/view?usp=share_link) (provided by the original repository of SAM) and the [LoRA checkpoint of SAMed](https://drive.google.com/file/d/1P0Bm-05l-rfeghbrT1B62v5eN-3A-uOr/view?usp=share_link). Put them in the `./checkpoints` folder.
3. Please download the [testset](https://drive.google.com/file/d/1RczbNSB37OzPseKJZ1tDxa5OO1IIICzK/view?usp=share_link) and put it in the ./testset folder. Then, unzip and deete this file.
4. Run this commend to test the performance of SAMed.
```bash
python test.py --is_savenii --output_dir <Your output directory>
```
If everything works, you can find the average DSC is 0.8188 (81.88) and HD is 20.64, which correspond to the Tab.1 of the paper. And check the test results in `<Your output directory>`.

What's more, we also provide the [SAMed_s model](https://drive.google.com/file/d/1rQM2md-h66RlRF3wC0m9N8aheOCvKfYv/view?usp=share_link), which utilizes LoRA to finetune the transformer blocks in image encoder and mask decoder. Compared with SAMed, SAMed_s has smaller model size but the performance also drops slightly. If you want to use this model, download and put it in the `./checkpoints_s` folder and run the below command to test its performance.
```bash
python test.py --is_savenii --output_dir <Your output directory> --lora_ckpt checkpoints_s/epoch_159.pth --module sam_lora_image_encoder_mask_decoder
```

## Training
We use 2 RTX 3090 GPUs for training.
1. Please download the processed [training set](https://drive.google.com/file/d/1zuOQRyfo0QYgjcU_uZs0X3LdCnAC2m3G/view?usp=share_link), whose resolution is `224x224`, and put it in <Your folder>. Then, unzip and delete this file. We also prepare the [training set](https://drive.google.com/file/d/1F42WMa80UpH98Pw95oAzYDmxAAO2ApYg/view?usp=share_link) with resolution `512x512` for reference, the `224x224` version of training set is downsampled from the `512x512` version.
2. Run this command to train SAMed.
```bash
python train.py --root_path <Your folder> --output <Your output path> --warmup --AdamW 
```
Check the results in `<Your output path>`.

## License
This work is licensed under MIT license. See the [LICENSE](LICENSE) for details.

## Citation
If our work inspires your research or some part of the codes are useful for your work, please cite our paper:
```bibtex
@article{samed,
  title={Customized Segment Anything Model for Medical Image Segmentation},
  author={Kaidong Zhang, and Dong Liu},
  year={2023}
}
```

## Contact
If you have any questions, please contact us via 
- richu@mail.ustc.edu.cn

## Acknowledgement
We appreciate the developers of [Segment Anything Model](https://github.com/facebookresearch/segment-anything) and the provider of the [Synapse multi-organ segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789).
