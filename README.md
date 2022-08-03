
## CA-SpaceNet: Counterfactual Analysis for 6D Pose Estimation in Space
This is an official pytorch implementation of the IROS 2022 paper **CA-SpaceNet: Counterfactual Analysis for 6D Pose Estimation in Space**. In this repository, we provide PyTorch code for training and testing the proposed CA-SpaceNet on the [Swisscube](https://github.com/cvlab-epfl/wide-depth-range-pose) and [SPEED](https://kelvins.esa.int/satellite-pose-estimation-challenge/data/) dataset.

[[arXiv](https://arxiv.org/abs/2207.07869)]

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
## Dataset
We provide two separate projects for CA-SpaceNet corresponding to the [SPEED](https://kelvins.esa.int/satellite-pose-estimation-challenge/data/) and [Swisscube](https://github.com/cvlab-epfl/wide-depth-range-pose) datasets.

**SwissCube**. The SwissCube dataset is a high fidelity dataset for 6D object pose estimation in space scenes. Accurate 3D meshes and physically-modeled astronomical objects are included in this dataset.
It contains 500 scenes, of which each scene has 100 image sequences, resulting in 50K images in total.
Consistent with SwissCube, 40K images are used for training, and the remaining 10K ones are used for testing.

**SPEED**. *The Spacecraft Pose Estimation Dataset*(SPEED) was firstly released on the Kelvins Satellite Pose Estimation Challenge in 2019.
It contains a large number of synthetic images and a small number of real satellite images.
The ground-truth labels of the testing set are not available because the competition is not ongoing.
Therefore, we divide the training set into two parts at random, 10K images for training and the remaining 2K ones for testing.

## Contact
If you have any questions about our work, please contact slwang19@fudan.edu.cn or sbwang21@m.fudan.edu.cn.

