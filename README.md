<h2 align="center">A deep learning method for building height estimation using high-resolution multi-view imagery over urban areas: A case study of 42 Chinese cities.</h2>

We introduce high-resolution ZY-3 multi-view images to estimate building height at a spatial resolution of 2.5 m. We propose a multi-spectral, multi-view, and multi-task deep network (called M3Net) for building height estimation, where ZY-3 multi-spectral and multi-view images are fused in a multi-task learning framework. By preprocessing the data from [Amap](https://amap.com), we obtained 4723 samples from the 42 cities (Table 1), and randomly selected 70%, 10%, and 20% of them for training, validation, and testing, respectively. Paper link ([website](https://www.sciencedirect.com/science/article/pii/S0034425721003102))

<h5 align="right">by Yinxia Cao, Xin Huang </h5>

---------------------


## Getting Started

#### Requirements:
- pytorch >= 1.8.0 (lower version can also work)
- python >=3.6

### Prepare the training set

See the sample directory. Due to the copyright problem, the whole dataset is not available publicly now.
However, the reference height data from Amap can be accessible for research use. Here is the download [link](https://pan.baidu.com/s/1bBTvZcPM6PeOXxxW3j_jOg) and extraction code is 4gn2 ). The provided data is original one, and preprocessing is needed before use.

### Train the height model
#### 1. Prepare your dataset
#### 2. edit data path
```
python train_zy3bh_tlcnetU_loss.py
```

#### 3. Evaluate on test set
see the pretrained model in directory runs/
```
python evaluate.py
```

If there is any issue, please feel free to contact me (email: yinxcao@163.com or yinxcao@whu.edu.cn).
## Citation

If you find this repo useful for your research, please consider citing the paper
```
@article{cao2021deep,
  title={A deep learning method for building height estimation using high-resolution multi-view imagery over urban areas: A case study of 42 Chinese cities},
  author={Cao, Yinxia and Huang, Xin},
  journal={Remote Sensing of Environment},
  volume={264},
  pages={112590},
  year={2021},
  publisher={Elsevier}
}
```
## Acknowledgement
```
@article{mshahsemseg,
    Author = {Meet P Shah},
    Title = {Semantic Segmentation Architectures Implemented in PyTorch.},
    Journal = {https://github.com/meetshah1995/pytorch-semseg},
    Year = {2017}
}
```
