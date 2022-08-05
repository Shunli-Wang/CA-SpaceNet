
## CA-SpaceNet on the SwissCube dataset
This is an official pytorch implementation of our IROS 2022 paper **CA-SpaceNet: Counterfactual Analysis for 6D Pose Estimation in Space**. In this repository, we provide PyTorch code for training and testing our proposed CA-SpaceNet on the challenging SwissCube dataset.

## Model Zoo
We provide CA-SpaceNet pretrained on the [SwissCube](https://github.com/cvlab-epfl/wide-depth-range-pose) dataset. WDR_78.78.pth is the weight obtained by training the WDR model on the Swisscube dataset. CA-SpaceNet trained with WDR_78.78.pth as the pre-training model, then outperforms state-of-the-arts on the challenging SwissCube dataset.

<table>
  <tr align="center"><td><b>Name</b></td><td><b>Dataset</b></td><td><b>Near</b></td><td><b>Medium</b></td><td><b>Far</b></td><td><b>All</b></td><td><b>Url</b></td></tr>
   <tr align="center"><td>WDR</td><td>Swisscube</td><td>92.37</td><td>84.16</td><td>61.27</td><td>78.78</td>
    <td><a href='https://pan.baidu.com/s/1_altEartEv2DXXbkW62h6Q'>BaiduNetDisk</a> [0am6] or <a href='https://drive.google.com/file/d/1QyRlulJ9u8WZgD7b3l7uNpw4bSf2PeYe/view?usp=sharing'>Google Drive</a></td>
  </tr>
  <tr align="center"><td>CA-SpaceNet</td><td>Swisscube</td><td>91.01</td><td>86.32</td><td>61.72</td><td>79.39</td>
    <td><a href='https://pan.baidu.com/s/1fZq9dxhKrJ5JgIlfip5kfQ'>BaiduNetDisk</a> [cwhi] or <a href='https://drive.google.com/file/d/1g3dEHI0GprakUe5jZk_GDoo2ukiGJRgb/view?usp=sharing'>Google Drive</a></td>
  </tr>
</table>

## User Guide

**1\. Clone this repository.**
```bash
git clone https://github.com/Shunli-Wang/CA-SpaceNet.git ./CA-SpaceNet
cd ./CA-SpaceNet/CA-SpaceNet-on-Swisscube
```

**2\. Create conda env.**
```bash
conda create -n CA-SpaceNet python=3.9
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

**3\. Download the Dataset & Create a soft link.**

 Download the [SwissCube](https://github.com/cvlab-epfl/wide-depth-range-pose) dataset and put it in `/PATH/TO/SwissCube_1.0`. Then create a soft link to this dataset. Note that the name of the dataset `SwissCube_1.0` is erased here.
```bash
ln -s /PATH/TO ./data
```
The downloading produce of pre-trained weights of DarkNet maybe extremely slow. So we'd better download the pth file manually and put it to predetermined location. In this way, the program will skip the automatic download process. 
```bash
unzip darknet53-0564-b36bef6b.pth.zip && rm darknet53-0564-b36bef6b.pth.zip
mv darknet53-0564-b36bef6b.pth ~/.torch/model/
```

**4\. Training & Testing**

Training CA-SpaceNet requires loading the pre-training weight of the WDR model.

#### Training
(1) If you want to train a CA-SpaceNet model, like the following:
```bash
1. download the WDR_78.78.pth file to the 'working_dirs' folder.
2. CUDA_VISIBLE_DEVICES=0 python3 train.py --weight_file ./working_dirs/WDR_78.78.pth
```
(2) If you want to train a CA-SpaceNet model from scratch, like the following:
```bash
1. CUDA_VISIBLE_DEVICES=0 python3 train_WDR.py
2. CUDA_VISIBLE_DEVICES=0 python3 train.py --weight_file ./working_dirs/WDR.pth # './working_dirs/WDR.pth' is the weight obtained by executing train_WDR.py.
```
#### Testing

After finishing the training phase, an experiment file will be created under `./working_dirs/swisscube`. For example, '20220311_123010'.

(1) Modify the weight path specified by the following command. Use this experiment number to replace the number in the original command. Then run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python3 test.py --weight_file 'working_dirs/swisscube/20211230_180655/final.pth'
```
(2) Or you can simply download our weight to start testing, like the following:
```
1. download the CA_79.39.pth file to the 'working_dirs' folder.
2. CUDA_VISIBLE_DEVICES=0 python3 test.py --weight_file 'working_dirs/CA_79.39.pth'
```

**5\. Visualization**
In order to better explore the performance of the model, this project provides visualization function. All prediction corners and 6D poses can be visualized by uncommenting the commands in `valid()` function of `train.py`.

## Cite
```
@article{CA-SpaceNet2022,
  title={CA-SpaceNet: Counterfactual Analysis for 6D Pose Estimation in Space},
  author={Wang, Shunli and Wang, Shuaibing and Jiao, Bo and Yang, Dingkang and Su, Liuzhen and Zhai, Peng and Chen, Chixiao and Zhang, Lihua},
  journal={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year={2022}
}
```

## Contact
If you have any questions about our work, please contact slwang19@fudan.edu.cn or sbwang21@m.fudan.edu.cn.

