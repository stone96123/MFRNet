![Python ==3.8](https://img.shields.io/badge/Python-==3.8-yellow.svg)
![PyTorch ==1.12.0](https://img.shields.io/badge/PyTorch-==1.12.0-blue.svg)

# [ICML2025] Multi-Modal Object Re-Identification via Sparse Mixture-of-Experts
The official repository for Multi-Modal Object Re-Identification via Sparse Mixture-of-Experts [[pdf]](https://arxiv.org/pdf/?.pdf)

### Prepare Datasets

```bash
mkdir data
```
Download the person datasets [RGBNT201](https://drive.google.com/drive/folders/1EscBadX-wMAT56_It5lXY-S3-b5nK1wH), [RGBNT100](https://pan.baidu.com/s/1xqqh7N4Lctm3RcUdskG0Ug code：rjin), and the [MSVR310](https://drive.google.com/file/d/1IxI-fGiluPO_Ies6YjDHeTEuVYhFdYwD/view?usp=drive_link).

### Installation

```bash
pip install -r requirements.txt
```

### Prepare ViT Pre-trained Models

You need to download the pretrained CLIP model : [ViT-B-16](https://pan.baidu.com/s/1YPhaL0YgpI-TQ_pSzXHRKw Code：52fu)

## Training

You can train the MFRNet with:

```bash
python train_net.py --config_file configs/RGBNT201/MFRNet.yml
```
**Some examples:**
```bash
python train_net.py --config_file configs/RGBNT201/MFRNet.yml
```

1. The device ID to be used can be set in config/defaults.py

2. If you need to train on the RGBNT100 and MSVR310 datasets, please ensure the corresponding path is modified accordingly.


## Evaluation

```bash
python test_net.py --config_file 'choose which config to test' --model_path 'your path of trained checkpoints'
```

**Some examples:**
```bash
python test_net.py --config_file configs/MSVR310/MFRNet.yml --model_path MSVR310_MFRNetbest.pth
```

#### Results
|  Dataset  | Rank@1 | mAP  | Model |
|:---------:|:------:|:----:| :------: |
| RGBNT201  |  83.6  | 80.7 | [model](https://drive.google.com) |
| RGBNT100  |  97.4  | 88.2 | [model](https://drive.google.com) |
| MSVR310   |  64.8  | 50.6 | [model](https://drive.google.com) |


## Citation
Please kindly cite this paper in your publications if it helps your research:
```bash
@inproceedings{fengmulti,
  title={Multi-Modal Object Re-identification via Sparse Mixture-of-Experts},
  author={Feng, Yingying and Li, Jie and Xie, Chi and Tan, Lei and Ji, Jiayi},
  booktitle={Forty-second International Conference on Machine Learning}
}
```

## Acknowledgement
Our code is based on [TOP-ReID](https://github.com/924973292/TOP-ReID)[1]

## References
[1]Wang Yuhao, Liu Xuehu, Zhang Pingping, Lu Hu, Tu Zhengzheng, Lu Huchuan. 2024. TOP-ReID: Multi-Spectral Object Re-identification with Token Permutation. Proceedings of the AAAI Conference on Artificial Intelligence. 38. 5758-5766.

## Contact

If you have any question, please feel free to contact us. E-mail: [tanlei@stu.xmu.edu.cn](mailto:tanlei@stu.xmu.edu.cn)
