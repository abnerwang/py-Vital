# py-VITAL
by [Xiaoping Wang](http://blog.keeplearning.group/about/).  
## Introduction
Python (PyTorch) implementation of VITAL tracker. VITAL is a great tracker invented by Song, Yibing and Ma, Chao and et al. This implementation is based on [py-MDNet](https://github.com/HyeonseobNam/py-MDNet), it is implemented by Hyeonseob Nam and Bohyung Han. Thanks to all of them.  

### [[Project]](https://ybsong00.github.io/cvpr18_tracking/index.html) [[Paper]](https://arxiv.org/pdf/1804.04273.pdf) [[Matlab code]](https://github.com/ybsong00/Vital_release)  

If you want this code for personal use, please cite:   

    @InProceedings{nam2016mdnet,
    author = {Nam, Hyeonseob and Han, Bohyung},
    title = {Learning Multi-Domain Convolutional Neural Networks for Visual Tracking},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2016}
    }  
    
    @inproceedings{song-cvpr18-VITAL,
    author = {Song, Yibing and Ma, Chao and Wu, Xiaohe and Gong, Lijun and Bao, Linchao and Zuo, Wangmeng and Shen, Chunhua and Lau, Rynson and Yang, Ming-Hsuan}, 
    title = {VITAL: VIsual Tracking via Adversarial Learning}, 
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},    
    year = {2018},
    }  
      
    @inproceedings{xpwang-VITAL-PyTorch,
    author = {Xiaoping Wang}, 
    title = {VITAL: VIsual Tracking via Adversarial Learning}, 
    booktitle = {VITAL tracker implemented by PyTorch}, 
    month = {March},
    year = {2019},
    }  

## Prerequisites
- python 3.6+
- opencv 3.0+
- [PyTorch 1.0+](http://pytorch.org/) and its dependencies

## Usage

### Tracking
```bash
 python tracking/run_tracker.py -s DragonBaby [-d (display fig)] [-f (save fig)]
```
 - You can provide a sequence configuration in two ways (see tracking/gen_config.py):
   - ```python tracking/run_tracker.py -s [seq name]```
   - ```python tracking/run_tracker.py -j [json path]```

### Pretraining
 - Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat"
 - Pretraining on VOT-OTB
   - Download [VOT](http://www.votchallenge.net/) datasets into "datasets/VOT/vot201x"
    ``` bash
     python pretrain/prepro_vot.py
     python pretrain/train_mdnet.py -d vot
    ```
 - Pretraining on ImageNet-VID
   - Download ImageNet-VID dataset into "datasets/ILSVRC"
    ``` bash
     python pretrain/prepro_imagenet.py
     python pretrain/train_mdnet.py -d imagenet
    ```
