# SAR-AD-BagNet
  The model and code for SAR-BagNet can be found at https://github.com/xwhu002/SAR-BagNet
## Citation
```
@article{li2022sar,
  title={SAR-BagNet: An Ante-hoc Interpretable Recognition Model Based on Deep Network for SAR Image},
  author={Li, Peng and Feng, Cunqian and Hu, Xiaowei and Tang, Zixiang},
  journal={Remote Sensing},
  volume={14},
  number={9},
  pages={2150},
  year={2022},
  publisher={MDPI}
}
```

## Prerequisites
* Python (3.6+)
* Pytorch (0.4.1)
* CUDA
* numpy

## Interpretable SAR Image Recognition based on Adversarial Defense

### AT-based training
run AT_train.py

### TRADES-based training

run TRADES_train,py

## Results

Our model achieves the following performance on :

###Classification and robustness on MSTAR 10 class vehicle

| Model name         |    Accuracy     | 
| ------------------ |---------------- |
| 8/255 AT-based SAR-BagNet     |    99.30 %      | 
| 8/255 TR-based SAR-BagNet     |    98.93 %      | 
| 16/255 AT-based SAR-BagNet    |    99.18 %      | 
| 16/255 TR-based SAR-BagNet    |    98.72 %      |

 
