## CA-SpaceNet: Counterfactual Analysis for 6D Pose Estimation in Space
This is an official pytorch implementation of our IROS 2022 paper **CA-SpaceNet: Counterfactual Analysis for 6D Pose Estimation in Space**. In this repository, we provide PyTorch code for training and testing our proposed CA-SpaceNet on the SPEED dataset.

[[arXiv](https://arxiv.org/abs/2207.07869)]

![image](https://user-images.githubusercontent.com/51118126/181905495-c813ab75-a2c7-46c5-a19a-0896f426ce82.png)

If this repository is helpful to you, please star it. If you find our work useful in your research, please consider citing:
```bash
@article{wang2022spacenet,
  title={CA-SpaceNet: Counterfactual Analysis for 6D Pose Estimation in Space},
  author={Wang, Shunli and Wang, Shuaibing and Jiao, Bo and Yang, Dingkang and Su, Liuzhen and Zhai, Peng and Chen, Chixiao and Zhang, Lihua},
  journal={arXiv preprint arXiv:2207.07869},
  year={2022}
}
```

## Model Zoo
We provide CA-SpaceNet and a replicated version of the WDR pretrained on the [SPEED](https://kelvins.esa.int/satellite-pose-estimation-challenge/data/) dataset. WDR_0.0400.pth is the weight obtained by training the WDR model on the SPEED dataset. CA-SpaceNet trained with WDR_0.0400.pth as the pre-training model, then achieves competitive results on the SPEED dataset.
<div class="center">
<table>
  <tr><td>name</td><td>dataset</td><td>S_R_m</td><td>S_t_m</td><td>S_total_m</td><td align="center">url</td></tr>
 
  <tr><td>WDR</td><td>SPEED</td><td>0.029241</td><td>0.010777</td><td>0.040018</td>
    <td><a href='https://pan.baidu.com/s/1eOK3D6D_tlGFQe6w0sh2fA'>BaiduNetDisk</a> [nxcx] or <a href='https://drive.google.com/file/d/1FXPuPOwyxbxomPo9mmjVIluEtrjjSoh1/view?usp=sharing'>Google Drive</a></td>
  </tr>
  
   <tr><td>CA-SpaceNet</td><td>SPEED</td><td>0.029006</td><td>0.009446</td><td>0.038451</td>
    <td><a href='https://pan.baidu.com/s/1JGLieEqzww1T4uinp2EkRA'>BaiduNetDisk</a> [pqtl] or <a href='https://drive.google.com/file/d/1_8OgWWRDsbSz3QDnhlez7mycWQgz5gHo/view?usp=sharing'>Google Drive</a></td>
  </tr>
</table>
</div>

## User Guide

**1\. Clone this repository.**
```bash
git clone -b SPEED_pose https://github.com/Shunli-Wang/CA-SpaceNet.git ./CA-SpaceNet
cd ./CA-SpaceNet
```

**2\. Create conda env.**

```bash
conda create -n CA-SpaceNet python=3.9
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

**3\. Download the Dataset & Create a soft link.**

 Download the `SPEED dataset`[1] and put it in `/PATH/TO/SPEED`. Then create a soft link to this dataset. Note that the name of the dataset `SPEED` is erased here.
```bash
ln -s /PATH/TO ./data
```
The downloading produce of pre-trained weights of DarkNet maybe extremely slow. So we'd better download the pth file manually and put it to predetermined location. In this way, the program will skip the automatic download process. 
```bash
unzip darknet53-0564-b36bef6b.pth.zip && rm darknet53-0564-b36bef6b.pth.zip
mv darknet53-0564-b36bef6b.pth ~/.torch/model/
```

**4\. Training & Testing**

Run the train.sh file:
```bash
sh ./train.sh
```
After finishing the training phase, an experiment file will be created under `./working_dirs/swisscube`. For example, '20220311_123010'.

Modify the instruction in `test.sh`. Use this experiment number to replace the number in the original command. Then run the test.sh file:
```bash
sh ./test.sh
```

**5\. Visualization**
In order to better explore the performance of the model, this project provides visualization function. All prediction corners and 6D poses can be visualized by uncommenting the commands in `valid()` function of `train.py`.


## Contact
If you have any questions about our work, please contact slwang19@fudan.edu.cn or sbwang21@m.fudan.edu.cn.
