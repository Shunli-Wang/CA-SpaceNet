## CA-SpaceNet on the SPEED dataset
This is an official pytorch implementation of our IROS 2022 paper **CA-SpaceNet: Counterfactual Analysis for 6D Pose Estimation in Space**. In this repository, we provide PyTorch code for training and testing our proposed CA-SpaceNet on the SPEED dataset.

## Model Zoo
We provide CA-SpaceNet and a replicated version of the WDR pretrained on the [SPEED](https://kelvins.esa.int/satellite-pose-estimation-challenge/data/) dataset. WDR_0.0400.pth is the weight obtained by training the WDR model on the SPEED dataset. CA-SpaceNet trained with WDR_0.0400.pth as the pre-training model, then achieves competitive results on the SPEED dataset.

<table>
    <tr align="center"><td><b>Name</b></td><td><b>Dataset</b></td><td><b>S_R_m</b></td><td><b>S_t_m</b></td><td><b>S_total_m</b></td><td><b>Url</b></td></tr> 
  <tr align="center"><td>WDR</td><td>SPEED</td><td>0.029241</td><td>0.010777</td><td>0.040018</td>
    <td><a href='https://pan.baidu.com/s/1eOK3D6D_tlGFQe6w0sh2fA'>BaiduNetDisk</a> [nxcx] or <a href='https://drive.google.com/file/d/1FXPuPOwyxbxomPo9mmjVIluEtrjjSoh1/view?usp=sharing'>Google Drive</a></td>
  </tr>
   <tr align="center"><td>CA-SpaceNet</td><td>SPEED</td><td>0.029006</td><td>0.009446</td><td>0.038451</td>
    <td><a href='https://pan.baidu.com/s/1JGLieEqzww1T4uinp2EkRA'>BaiduNetDisk</a> [pqtl] or <a href='https://drive.google.com/file/d/1_8OgWWRDsbSz3QDnhlez7mycWQgz5gHo/view?usp=sharing'>Google Drive</a></td>
  </tr>
</table>

## User Guide

**1\. Clone this repository.**
```bash
git clone https://github.com/Shunli-Wang/CA-SpaceNet.git ./CA-SpaceNet
cd ./CA-SpaceNet/CA-SpaceNet-on-SPEED
```

**2\. Create conda env.**
```bash
conda create -n CA-SpaceNet python=3.9
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

**3\. Download the Dataset & Create a soft link.**

 Download the [SPEED dataset](https://kelvins.esa.int/satellite-pose-estimation-challenge/data/) [1] and put it in `/PATH/TO/SPEED`. Then create a soft link to this dataset. Note that the name of the dataset `SPEED` is erased here.
```bash
ln -s /PATH/TO ./data
```

In addition, supplementary data for the SPEED dataset needs to be downloaded from <a href='https://pan.baidu.com/s/1i8I_JHqIukb_2v-EtZp5dQ'>BaiduNetDisk</a> [e20g].

The final data directory format is as follows:
```bash
├── boxes.json
├── images
│   ├── real
│   ├── real_test
│   ├── test
│   └── train
├── K.json
├── masks
│   └──  train
├── real.json
├── split
│   ├── trainf1.txt
│   ├── trainf2.txt
│   ├── trainf3.txt
│   ├── trainf4.txt
│   ├── trainf5.txt
│   ├── trainf6.txt
│   ├── validf1.txt
│   ├── validf2.txt
│   ├── validf3.txt
│   ├── validf4.txt
│   ├── validf5.txt
│   └── validf6.txt
├── testing.txt
├── test.json
├── training.txt
└── train.json
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
# download the WDR_0.0400.pth file to the 'working_dirs' folder.
CUDA_VISIBLE_DEVICES=0 python3 train.py --weight_file ./working_dirs/WDR_0.0400.pth
```
(2) If you want to train a CA-SpaceNet model from scratch, like the following:
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_WDR.py
CUDA_VISIBLE_DEVICES=0 python3 train.py --weight_file ./working_dirs/WDR.pth # './working_dirs/WDR.pth' is the weight obtained by executing train_WDR.py.
```
#### Testing

After finishing the training phase, an experiment file will be created under `./working_dirs/swisscube`. For example, '20220311_123010'.

(1) Modify the weight path specified by the following command. Use this experiment number to replace the number in the original command. Then run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python3 test.py --weight_file 'working_dirs/swisscube/20211230_180655/final.pth'
```
(2) Or you can simply download our weight to start testing, like the following:
```
1. download the CA_0.0385.pth file to the 'working_dirs' folder.
2. CUDA_VISIBLE_DEVICES=0 python3 test.py --weight_file 'working_dirs/CA_0.0385.pth'
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
