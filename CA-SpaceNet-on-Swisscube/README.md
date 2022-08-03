
## CA-SpaceNet on the SwissCube dataset
This is an official pytorch implementation of our IROS 2022 paper **CA-SpaceNet: Counterfactual Analysis for 6D Pose Estimation in Space**. In this repository, we provide PyTorch code for training and testing our proposed CA-SpaceNet on the challenging SwissCube dataset.

## Model Zoo
We provide CA-SpaceNet pretrained on the [SwissCube](https://github.com/cvlab-epfl/wide-depth-range-pose) dataset. WDR_78.78.pth is the weight obtained by training the WDR model on the Swisscube dataset. CA-SpaceNet trained with WDR_78.78.pth as the pre-training model, then outperforms state-of-the-arts on the challenging SwissCube dataset.
<!-- 这个地方表头首字母大写，加粗居中。 -->

<table>
  <tr><td>name</td><td>dataset</td><td>near</td><td>medium</td><td>far</td><td>all</td><td align="center">url</td></tr>
   <tr><td>WDR</td><td>Swisscube</td><td>92.37</td><td>84.16</td><td>61.27</td><td>78.78</td>
    <td><a href='https://pan.baidu.com/s/1_altEartEv2DXXbkW62h6Q'>BaiduNetDisk</a> [0am6] or <a href='https://drive.google.com/file/d/1QyRlulJ9u8WZgD7b3l7uNpw4bSf2PeYe/view?usp=sharing'>Google Drive</a></td>
  </tr>
  <tr><td>CA-SpaceNet</td><td>Swisscube</td><td>91.01</td><td>86.32</td><td>61.72</td><td>79.39</td>
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
<!-- 这里需要添加上一些说明：如果用户想基于7878预训练模型进行CA-SpaceNet模型的训练，则需要先下载好WDR模型，放到特定的位置，然后再运行这个sh文件 -->
Run the train.sh file:
```bash
sh ./train.sh
```

<!-- 测试有两种方法，一种是用户自己训练出一个CA-SpaceNet模型，则需要进行命令行修改，再运行sh文件；另一种是直接下载预训练模型，同样也是修改命令行再运行。 -->
After finishing the training phase, an experiment file will be created under `./working_dirs/swisscube`. For example, '20220311_123010'.

Modify the instruction in `test.sh`. Use this experiment number to replace the number in the original command. Then run the test.sh file:
```bash
sh ./test.sh
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

