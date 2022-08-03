
## CA-SpaceNet: Counterfactual Analysis for 6D Pose Estimation in Space
This is an official pytorch implementation of the IROS 2022 paper **CA-SpaceNet: Counterfactual Analysis for 6D Pose Estimation in Space**. In this repository, we provide PyTorch code for training and testing the proposed CA-SpaceNet on the [Swisscube](https://github.com/cvlab-epfl/wide-depth-range-pose) and [SPEED](https://kelvins.esa.int/satellite-pose-estimation-challenge/data/) dataset.

[[arXiv](https://arxiv.org/abs/2207.07869)][[supp video](https://www.youtube.com/watch?v=h-vzCdersVQ)]

![image](https://user-images.githubusercontent.com/51118126/181905495-c813ab75-a2c7-46c5-a19a-0896f426ce82.png)

If this repository is helpful to you, please star it. If you find our work useful in your research, please consider citing:
```bash
@article{CA-SpaceNet2022,
  title={CA-SpaceNet: Counterfactual Analysis for 6D Pose Estimation in Space},
  author={Wang, Shunli and Wang, Shuaibing and Jiao, Bo and Yang, Dingkang and Su, Liuzhen and Zhai, Peng and Chen, Chixiao and Zhang, Lihua},
  journal={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year={2022}
}
```
## Repository Architecture
In this repository, we provide two separate folders to store two sets of codes on the [Swisscube](https://github.com/cvlab-epfl/wide-depth-range-pose) dataset and the [SPEED](https://kelvins.esa.int/satellite-pose-estimation-challenge/data/) dataset, respectively.

- **./CA-SpaceNet-on-Swisscube/**: This folder contains all codes of the proposed CA-SpaceNet on the [Swisscube](https://github.com/cvlab-epfl/wide-depth-range-pose) dataset and pre-trained models. Please refer to [here](./CA-SpaceNet-on-Swisscube/README.md) for more details.

- **./CA-SpaceNet-on-SPEED/**: This folder contains all codes of the proposed CA-SpaceNet on the [SPEED](https://kelvins.esa.int/satellite-pose-estimation-challenge/data/) dataset and pre-trained models. Please refer to [here](./CA-SpaceNet-on-SPEED/README.md) for more details.

## Contact


## Contact
If you have any questions about our work, please contact slwang19@fudan.edu.cn or sbwang21@m.fudan.edu.cn.
