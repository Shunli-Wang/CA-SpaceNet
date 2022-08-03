from email.mime import image
import os
import json
import re
import cv2
import random
import numpy as np

import torch
from torch.utils.data import Dataset

from core.boxlist import BoxList
from core.poses import PoseAnnot

from core.utils import (
    load_bop_meshes,
    load_bbox_3d,
    get_single_bop_annotation
)

class BOP_Dataset_train(Dataset):
    def __init__(self, image_list_file, bbox_json, pose_json, transform, samples_count=0, training=True):
        # file list and data should be in the same directory
        dataDir = os.path.split(os.path.split(image_list_file)[0])[0]  # './data/SPEED/training.txt'
        with open(image_list_file, 'r') as f:
            self.img_files = f.readlines()
            self.img_files = [dataDir + '/' + x.strip() for x in self.img_files]
        # 
        rawSampleCount = len(self.img_files)
        if training and samples_count > 0:
            self.img_files = random.choices(self.img_files, k = samples_count)

        if training:
            random.shuffle(self.img_files)

        print("Number of samples: %d / %d" % (len(self.img_files), rawSampleCount))

        # Object dict
        self.objID_2_clsID = {'1':0}

        # 11 3D points
        self.bbox_3d = load_bbox_3d(bbox_json)

        # Load Ann file
        with open(pose_json, 'r') as f:
            self.pose_json = json.load(f)
        self.pose_json_idx = [self.pose_json[i]['filename'] for i in range(len(self.pose_json))]

        self.transformer = transform
        self.training = training

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        item = self.getitem1(index)
        while item is None:
            index = random.randint(0, len(self.img_files) - 1)
            item = self.getitem1(index)
        return item

    def getitem1(self, index):
        img_path = self.img_files[index]

        # Load image
        try:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # BGR(A)
            
            # load Mask 使用形态学操作将卫星部分进行扣除
            train_or_test=img_path.split('/')[-2]+'/'
            mask_path=img_path[:-26]+'masks/'+train_or_test+img_path[-13:-4]+'.png'
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # BGR(A)
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.dilate(mask, kernel)  # 膨胀操作
            mask = cv2.bitwise_not(mask)
            masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
            
            # cv2.imwrite('./masked/' + img_path[-13:-4] + '_.jpg', img)
            # cv2.imwrite('./masked/' + img_path[-13:-4] + '.jpg', masked)

            if img is None:
                raise RuntimeError('load image error')
            # 
            if img.dtype == np.uint16:
                img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0)).astype(np.uint8)
            # 
            if len(img.shape) == 2:
                # convert gray to 3 channels
                img = np.repeat(img.reshape(img.shape[0], img.shape[1], 1), 3, axis=2)
            # elif img.shape[2] == 3:
            #     # add an alpha channel
            #     img = np.concatenate((img, np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8)*255), axis=-1)
            elif img.shape[2] == 4:
                # having alpha
                tmpBack = (img[:,:,3] == 0)
                img[:,:,0:3][tmpBack] = 255 # white background

            # padding image from (1200, 1920, 3) to (1920, 1920, 3)
            padd_0 = np.zeros((360, 1920, 3)).astype(np.uint8) 
            padd_1 = np.zeros((360, 1920, 3)).astype(np.uint8)  
            img = np.concatenate((padd_0, img), axis=0)
            img = np.concatenate((img, padd_1), axis=0)

            # padding masked
            padd_0 = np.zeros((360, 1920)).astype(np.uint8)
            padd_1 = np.zeros((360, 1920)).astype(np.uint8)
            # print(mask_vis.shape, padd.shape)
            masked = np.concatenate((padd_0, masked), axis=0)
            masked = np.concatenate((masked, padd_1), axis=0)

            # cv2.imwrite('./masked/' + img_path[-13:-4] + '_.jpg', img)
            # cv2.imwrite('./masked/' + img_path[-13:-4] + '.jpg', masked)

        except Exception as e:
            print(e)
            print('image %s not found' % img_path)
            return None

        # Load labels (BOP format)
        height, width, _ = img.shape
        # ########################## To Do: 此处需要完成数据集信息的加载(考虑使用投影方法生成GT点，而不是依靠matlab中的生成结果-->为了给以后的数据生成打基础)
        K, merged_mask, class_ids, rotations, translations = get_single_bop_annotation([height, width], img_path, self.pose_json, self.pose_json_idx, self.objID_2_clsID)
        # cv2.imwrite('./masked/' + img_path[-13:-4] + '__.jpg', merged_mask * 255)

        # get (raw) image meta info
        meta_info = {
            'path': img_path,
            'K': K,
            'width': width,
            'height': height,
            'class_ids': class_ids,
            'rotations': rotations,
            'translations': translations
        }

        # ########################## To Do: 姿势标签生成
        target_ = PoseAnnot(self.bbox_3d, K, merged_mask, class_ids, rotations, translations, width, height)

        # ########################## To Do: 1.完成图像的黑底色填充; 2.保证transformer不会对K产生影响
        # transformation
        img, target = self.transformer(img, target_)
        masked, _ = self.transformer(masked, target_)
        
        target = target.remove_invalids(min_area = 10)
        if self.training and len(target) == 0:
            print("WARNING: skipped a sample without any targets")
            return None

        return img, masked, target, meta_info


class BOP_Dataset_test(Dataset):
    def __init__(self, image_list_file, bbox_json, pose_json, transform, samples_count=0, training=True):
        # file list and data should be in the same directory
        dataDir = os.path.split(os.path.split(image_list_file)[0])[0]  # './data/SPEED/training.txt'
        with open(image_list_file, 'r') as f:
            self.img_files = f.readlines()
            self.img_files = [dataDir + '/' + x.strip() for x in self.img_files]
        # 
        rawSampleCount = len(self.img_files)
        if training and samples_count > 0:
            self.img_files = random.choices(self.img_files, k = samples_count)

        if training:
            random.shuffle(self.img_files)

        print("Number of samples: %d / %d" % (len(self.img_files), rawSampleCount))

        # Object dict
        self.objID_2_clsID = {'1':0}

        # 11 3D points
        self.bbox_3d = load_bbox_3d(bbox_json)

        # Load Ann file
        with open(pose_json, 'r') as f:
            self.pose_json = json.load(f)
        self.pose_json_idx = [self.pose_json[i]['filename'] for i in range(len(self.pose_json))]

        self.transformer = transform
        self.training = training

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        item = self.getitem1(index)
        while item is None:
            index = random.randint(0, len(self.img_files) - 1)
            item = self.getitem1(index)
        return item

    def getitem1(self, index):
        img_path = self.img_files[index]

        # Load image
        try:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # BGR(A)
            
            if img is None:
                raise RuntimeError('load image error')
            # 
            if img.dtype == np.uint16:
                img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0)).astype(np.uint8)
            # 
            if len(img.shape) == 2:
                # convert gray to 3 channels
                img = np.repeat(img.reshape(img.shape[0], img.shape[1], 1), 3, axis=2)
            # elif img.shape[2] == 3:
            #     # add an alpha channel
            #     img = np.concatenate((img, np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8)*255), axis=-1)
            elif img.shape[2] == 4:
                # having alpha
                tmpBack = (img[:,:,3] == 0)
                img[:,:,0:3][tmpBack] = 255 # white background
            # padding image from (1200, 1920, 3) to (1920, 1920, 3)
            padd_0 = np.zeros((360, 1920, 3)).astype(np.uint8) 
            padd_1 = np.zeros((360, 1920, 3)).astype(np.uint8)  
            img = np.concatenate((padd_0, img), axis=0)
            img = np.concatenate((img, padd_1), axis=0)
        except:
            print('image %s not found' % img_path)
            return None

        # Load labels (BOP format)
        height, width, _ = img.shape
        # ########################## To Do: 此处需要完成数据集信息的加载(考虑使用投影方法生成GT点，而不是依靠matlab中的生成结果-->为了给以后的数据生成打基础)
        K, merged_mask, class_ids, rotations, translations = get_single_bop_annotation([height, width], img_path, self.pose_json, self.pose_json_idx, self.objID_2_clsID)
        
        # get (raw) image meta info
        meta_info = {
            'path': img_path,
            'K': K,
            'width': width,
            'height': height,
            'class_ids': class_ids,
            'rotations': rotations,
            'translations': translations
        }

        # ########################## To Do: 姿势标签生成
        target = PoseAnnot(self.bbox_3d, K, merged_mask, class_ids, rotations, translations, width, height)

        # ########################## To Do: 1.完成图像的黑底色填充; 2.保证transformer不会对K产生影响
        # transformation
        img, target = self.transformer(img, target)
        
        target = target.remove_invalids(min_area = 10)
        if self.training and len(target) == 0:
            print("WARNING: skipped a sample without any targets")
            return None

        return img, target, meta_info

class ImageList:
    def __init__(self, tensors, sizes):
        self.tensors = tensors
        self.sizes = sizes

    def to(self, *args, **kwargs):
        tensor = self.tensors.to(*args, **kwargs)

        return ImageList(tensor, self.sizes)

def image_list(tensors, size_divisible=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        if max_size[1] % stride != 0:
            max_size[1] = (max_size[1] | (stride - 1)) + 1
        if max_size[2] % stride != 0:
            max_size[2] = (max_size[2] | (stride - 1)) + 1
        max_size = tuple(max_size)

    shape = (len(tensors),) + max_size
    batch = tensors[0].new(*shape).zero_()

    for img, pad_img in zip(tensors, batch):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    sizes = [img.shape[-2:] for img in tensors]

    return ImageList(batch, sizes)

def collate_fn_train(size_divisible):
    def collate_data(batch):
        batch = list(zip(*batch))
        imgs = image_list(batch[0], size_divisible)
        masked = image_list(batch[1], size_divisible)
        targets = batch[2]
        meta_infos = batch[3]
        return imgs, masked, targets, meta_infos

    return collate_data

def collate_fn_test(size_divisible):
    def collate_data(batch):
        batch = list(zip(*batch))
        imgs = image_list(batch[0], size_divisible)
        targets = batch[1]
        meta_infos = batch[2]
        return imgs, targets, meta_infos

    return collate_data