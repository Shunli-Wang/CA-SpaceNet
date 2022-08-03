import os

import numpy as np

import torch
from torch import nn
from core.boxlist import cat_boxlist, boxlist_iou

import math

INF = 100000000
     
class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-6

    def forward(self, out, target):
        n_class = out.shape[1]
        class_ids = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        p = torch.sigmoid(out)
        p = torch.clamp(p, self.eps, 1-self.eps) # for numerical stability

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)

        # print(term1.sum(), term2.sum())

        loss = (
            -(t == class_ids).float() * alpha * term1
            - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )

        return loss.sum()

def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

def concat_box_prediction_layers(pred_cls, pred_reg):
    """ Transform outputs of the network to list form and get GT & loss.
    # pred_cls: Grid score of every layer
    # torch.Size([8, 1, 64, 64])
    # torch.Size([8, 1, 32, 32])
    # torch.Size([8, 1, 16, 16])
    # torch.Size([8, 1, 8, 8])
    # torch.Size([8, 1, 4, 4])

    # pred_reg: offset of every layer
    # torch.Size([8, 16, 64, 64]) 
    # torch.Size([8, 16, 32, 32]) 
    # torch.Size([8, 16, 16, 16]) 
    # torch.Size([8, 16, 8, 8])
    # torch.Size([8, 16, 4, 4])
    """
    pred_cls_flattened = []
    pred_reg_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the pred_reg
    for pred_cls_per_level, pred_reg_per_level in zip(pred_cls, pred_reg):
        N, AxC, H, W = pred_cls_per_level.shape
        Cx16 = pred_reg_per_level.shape[1]
        C = Cx16 // 16
        A = 1
        pred_cls_per_level = permute_and_flatten(pred_cls_per_level, N, A, C, H, W)  # N=8, A=1, C=1, H, W
        pred_cls_flattened.append(pred_cls_per_level)

        pred_reg_per_level = permute_and_flatten(pred_reg_per_level, N, A, (C*16), H, W)  # N=8, A=1, (C*16)=16, H, W
        pred_reg_flattened.append(pred_reg_per_level)

    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    pred_cls = cat(pred_cls_flattened, dim=1).reshape(-1, C)  # torch.Size([43648, 1])
    pred_reg = cat(pred_reg_flattened, dim=1).reshape(-1, C*16)  # torch.Size([43648, 16])
    return pred_cls, pred_reg

class PoseLoss(object):
    def __init__(self, gamma, alpha, anchor_sizes, anchor_strides, positive_num, positive_lambda,
                    loss_weight_cls, loss_weight_reg, internal_K, diameters, target_coder):
        # setting Focal loss.
        self.cls_loss_func = SigmoidFocalLoss(gamma, alpha)  # 2.0 0.25
        # self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        # self.matcher = Matcher(fg_iou_threshold, bg_iou_threshold, True)

        self.anchor_sizes = anchor_sizes        # [32, 64, 128, 256, 512]
        self.anchor_strides = anchor_strides    # [8, 16, 32, 64, 128]
        self.positive_num = positive_num        # 10
        self.positive_lambda = positive_lambda  # 1.0
        self.loss_weight_cls = loss_weight_cls  # 0.01
        self.loss_weight_reg = loss_weight_reg  # 0.1
        self.internal_K = internal_K   
        self.target_coder = target_coder
        self.diameters = diameters              # [178.46]

    def ObjectSpaceLoss(self, pred, target_3D_in_camera_frame, cls_labels, anchors, weight=None):
        # pred: torch.Size([78?, 16])
        # target_3D_in_camera_frame: torch.Size([78?, 8, 3])
        # cls_labels: torch.Size([78?])
        # anchors: torch.Size([78?, 4])

        if not isinstance(self.diameters, torch.Tensor):
            self.diameters = torch.FloatTensor(self.diameters).to(device=pred.device).view(-1)
        
        # Extend diameter
        diameter_ext = self.diameters[cls_labels.view(-1,1).repeat(1, 8*3).view(-1, 3, 1)]  # [78? *8,3,1]

        ################################ Process output of the network ##################################
        cellNum = pred.shape[0]  # Selected numbers
        pred_filtered = pred.view(cellNum, -1, 16)[torch.arange(cellNum), cls_labels]  # Only filter out one class. torch.Size([78?, 16])

        # Decoder: transform the output of the network (anchor-based presentation) into real number.
        pred_xy = self.target_coder.decode(pred_filtered, anchors)  # torch.Size([78?, 16])
        # target_xy = self.target_coder.decode(target, anchors)
        pred_xy = pred_xy.view(-1,2,8).transpose(1,2).contiguous().view(-1,2)  # shape transform: [78?*8, 2]
        
        # construct normalized 2d (project Pred-2D-points to ray in 3D space)  size:[78?*8, 2]
        B = torch.inverse(self.internal_K).mm(torch.cat((pred_xy.t(), torch.ones_like(pred_xy[:,0]).view(1,-1)), dim=0)).t()  # normlized coordinates: cat([2, 78?*8], [1, 78?*8]) ==> [3, 78?*8]
        # K^(-1) mm* ([3, 78?*8])

        ################################ Process GT info ##################################
        # compute projection matrices P
        P = torch.bmm(B.view(-1, 3, 1), B.view(-1, 1, 3)) / torch.bmm(B.view(-1, 1, 3), B.view(-1, 3, 1))  # torch.Size([616, 3, 3])

        # Real 3D position in camera frame.
        target_3D_in_camera_frame = target_3D_in_camera_frame.view(-1, 3, 1)  # torch.Size([78?, 3, 1])  [X,Y,Z] with depth
        px = torch.bmm(P, target_3D_in_camera_frame) # [78? *8, 3, 3] mm* [78? *8, 3, 1] ==> torch.Size([78? *8, 3, 1])

        target_3D_in_camera_frame = target_3D_in_camera_frame / diameter_ext
        px = px / diameter_ext
        scaling_factor = 50 # 0.02d

        losses = nn.SmoothL1Loss(reduction='none')(scaling_factor * px, scaling_factor * target_3D_in_camera_frame).view(cellNum, -1).mean(dim=1)
        losses = losses / scaling_factor

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

    def prepare_targets(self, targets, anchors):
        cls_labels = []
        reg_targets = []
        aux_raw_boxes = []
        aux_3D_in_camera_frame = []
        level_cnt = len(anchors[0])  # anchors [8,5]. five layers in total.

        for im_i in range(len(targets)):  # 8 images
            pose_targets_per_im = targets[im_i]  # Fetch a GT label.

            # Get Bbox info in xyxy
            bbox_targets_per_im = pose_targets_per_im.to_object_boxlist()
            assert bbox_targets_per_im.mode == "xyxy"
            bboxes_per_im = bbox_targets_per_im.bbox  # tensor([[176.5559, 228.3121, 228.6003, 287.6865]])

            # cls
            labels_per_im = pose_targets_per_im.class_ids + 1

            # anchors of every image
            anchors_per_im = cat_boxlist(anchors[im_i])  # Put all anchors to a single list: BoxList(num_boxes=5456, image_width=512, image_height=512, mode=xyxy) 64**2+32**2+16**2+8**2+4**2
            num_gt = bboxes_per_im.shape[0]  # 1
            assert(level_cnt == len(anchors[im_i]))  # level number check

            # Fetch R & T & Mask info of every image.
            rotations_per_im = pose_targets_per_im.rotations  # R
            translations_per_im = pose_targets_per_im.translations  # T
            mask_per_im = pose_targets_per_im.mask  # Mask  torch.Size([512, 512])

            # Anchor size of every layer
            anchor_sizes_per_level_interest = self.anchor_sizes[:level_cnt]  # [32, 64, 128, 256, 512]
            anchor_strides_per_level_interst = self.anchor_strides[:level_cnt] # [8, 16, 32, 64, 128]

            # Box Δx Δy = pixel number
            gt_object_sizes = bbox_targets_per_im.box_span()

            # anchors number of every layer
            num_anchors_per_level = [len(anchors_per_level.bbox) for anchors_per_level in anchors[im_i]]  # [4096, 1024, 256, 64, 16]
            # center coordinates of every anchor
            anchors_cx_per_im = (anchors_per_im.bbox[:, 2] + anchors_per_im.bbox[:, 0]) / 2.0  # tensor([  4.,  12.,  20.,  ..., 192., 320., 448.], device='cuda:0') [5456]
            anchors_cy_per_im = (anchors_per_im.bbox[:, 3] + anchors_per_im.bbox[:, 1]) / 2.0  # tensor([  4.,   4.,   4.,  ..., 448., 448., 448.], device='cuda:0') [5456]
            anchors_cx_per_im = torch.clamp(anchors_cx_per_im, min = 0, max = mask_per_im.shape[1] - 1).long()
            anchors_cy_per_im = torch.clamp(anchors_cy_per_im, min = 0, max = mask_per_im.shape[0] - 1).long()

            # Only anchors in mask are selected during training.
            mask_at_anchors = mask_per_im[anchors_cy_per_im, anchors_cx_per_im]
            mask_labels = []
            for gt_i in range(num_gt):  # 1. the goal of for loop is to process multi class
                valid_mask = (mask_at_anchors == (gt_i+1))
                mask_labels.append(valid_mask)
            mask_labels = torch.stack(mask_labels).t()
            mask_labels = mask_labels.long()  # torch.Size([5456, 1])

            ##################################### Sampling function #####################################
            # random selecting candidates from each level first
            gt_sz = gt_object_sizes.view(1,-1).repeat(level_cnt,1)  # Real pixel size of the target: tensor([[126.9868], [126.9868], [126.9868], [126.9868], [126.9868]])
            lv_sz = torch.FloatTensor(anchor_sizes_per_level_interest).type_as(gt_sz)  # anchor size & extention: tensor([[ 32.], [ 64.], [128.], [256.], [512.]])
            lv_sz = lv_sz.view(-1,1).repeat(1,num_gt)
            dk = torch.log2(gt_sz/lv_sz).abs()  # Compute size and log, abs
            nk = torch.exp(-self.positive_lambda * (dk * dk))  # exp(-1*dk^2)
            nk = self.positive_num * nk / nk.sum(0, keepdim=True)
            nk = (nk + 0.5).int()  # Number of selected samples in every layer. tensor([[0], [1], [6], [3], [0]], device='cuda:0', dtype=torch.int32)

            # Fetch specific number of points in every layer.
            candidate_idxs = [[] for i in range(num_gt)]
            start_idx = 0
            for level in range(level_cnt):
                end_idx = start_idx + num_anchors_per_level[level]  # sequencal order
                is_in_mask_per_level = mask_labels[start_idx:end_idx, :]  # Select all masks
                # 
                for gt_i in range(num_gt):  # multi classes
                    posi_num = nk[level][gt_i]  # Selected numbers

                    valid_pos = is_in_mask_per_level[:, gt_i].nonzero().view(-1)  # locate in all non-zero positions.
                    posi_num = min(posi_num, len(valid_pos))
                    # rand_idx = torch.randint(0, len(valid_pos), (int(posi_num),)) # randoms with replacement
                    rand_idx = torch.randperm(len(valid_pos))[:posi_num]        # randoms without replacement
                    candi_pos = valid_pos[rand_idx] + start_idx                 # Compute global location
                    candidate_idxs[gt_i].append(candi_pos)                      # Record position
                # 
                start_idx = end_idx  # Update the start in the next level.

            # flagging selected positions
            roi = torch.full_like(mask_labels, -INF)  # torch.Size([5456, 1])
            for gt_i in range(num_gt):
                tmp_idx = torch.cat(candidate_idxs[gt_i], dim=0)
                # candidate_idxs:
                # [[tensor([2713, 2901, 2841, 2777, 2904, 2903], device='cuda:0'), 
                # tensor([4812, 4780], device='cuda:0'), 
                # tensor([], device='cuda:0', dtype=torch.int64), 
                # tensor([], device='cuda:0', dtype=torch.int64), 
                # tensor([], device='cuda:0', dtype=torch.int64)]]
                #   ===>
                # tensor([4621, 4719, 4588, 4590, 5239, 5224, 5222, 5238, 5255, 5396])
                roi[tmp_idx, gt_i] = 1  # Selected pose "1"

            # Background setting
            anchors_to_gt_values, anchors_to_gt_indexs = roi.max(dim=1)
            cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
            cls_labels_per_im[anchors_to_gt_values == -INF] = 0 # background setting

            # In mask but not background
            mask_visibilities, _ = mask_labels.max(dim=1)
            # logical_and, introduced only after pytorch 1.5
            # ignored_indexs = torch.logical_and(mask_visibilities==1, cls_labels_per_im==0)
            ignored_indexs = (mask_visibilities == 1) * (cls_labels_per_im == 0)  # Background and in mask
            cls_labels_per_im[ignored_indexs] = -1 # positions within mask but not selected will not be touched

            ###################################### Generate 3D labels #####################################
            # 
            matched_boxes = bboxes_per_im[anchors_to_gt_indexs]                 # torch.Size([1, 4]) ==> torch.Size([5456, 4])
            matched_classes = (labels_per_im - 1)[anchors_to_gt_indexs]         # torch.Size([1]) ==> torch.Size([5456])
            matched_rotations = rotations_per_im[anchors_to_gt_indexs]          # torch.Size([1, 3, 3]) ==> torch.Size([5456, 3, 3])
            matched_translations = translations_per_im[anchors_to_gt_indexs]    # torch.Size([1, 3, 1]) ==> torch.Size([5456, 3, 1])

            # 
            matched_3Ds = pose_targets_per_im.keypoints_3d[matched_classes]  # torch.Size([5456, 8, 3])
            # a=[-52.1000, -52.1000, -52.1000, -52.1000,  52.1000,  52.1000,  52.1000, 52.1000]'
            # b=[-55.4000, -55.4000,  53.9000,  53.9000, -55.4000, -55.4000,  53.9000, 53.9000]'
            # c=[-56.7500,  58.0000, -56.7500,  58.0000, -56.7500,  58.0000, -56.7500, 58.0000]'

            # matched_Ks = pose_targets_per_im.K.repeat(matched_classes.shape[0], 1, 1)
            # TODO
            # assert equals self K

            if not isinstance(self.internal_K, torch.Tensor):
                self.internal_K = torch.FloatTensor(self.internal_K).to(device=matched_3Ds.device).view(3, 3)

            # Project GT 3D points into image frame and use anchor-based method to present.
            reg_targets_per_im = self.target_coder.encode(
                self.internal_K,        # torch.Size([3, 3]) 
                matched_3Ds,            # torch.Size([5456, 8, 3])
                matched_rotations,      # torch.Size([5456, 3, 3])
                matched_translations,   # torch.Size([5456, 3, 1])
                anchors_per_im.bbox     # torch.Size([5456, 4])
                )         

            cls_labels.append(cls_labels_per_im)    # torch.Size([5456]) 0-background ; 1-in mask but not background ; 1-in mask and selected
            reg_targets.append(reg_targets_per_im)  # torch.Size([5456, 16])  
            aux_raw_boxes.append(matched_boxes)     # torch.Size([5456, 4])

            # In camera frame, the projection of GT 3D points.
            matched_3D_in_camera_frame = torch.bmm(matched_rotations, matched_3Ds.transpose(1, 2)) + matched_translations  # [5456, 3, 3] * [5456, 3, 8] + [5456, 3, 1]
            aux_3D_in_camera_frame.append(matched_3D_in_camera_frame.transpose(1, 2))  # Size([5456, 8, 3]) --> Size([5456, 3, 8])

        return cls_labels, reg_targets, aux_raw_boxes, aux_3D_in_camera_frame

    def __call__(self, pred_cls, pred_reg, targets, anchors):

        ###################################### Process label info & Extension #####################################
        labels, reg_targets, aux_raw_boxes, aux_3D_in_camera_frame = self.prepare_targets(targets, anchors)

        labels_flatten = torch.cat(labels, dim=0)  # torch.Size([43648]) Contains all cls infor of all anchors: 0-background ; 1-in mask but not background ; 1-in mask and selected
        reg_targets_flatten = torch.cat(reg_targets, dim=0)  # torch.Size([43648, 16])
        aux_raw_boxes_flatten = torch.cat(aux_raw_boxes, dim=0)  # torch.Size([43648, 4])
        aux_3D_in_camera_frame_flatten = torch.cat(aux_3D_in_camera_frame, dim=0)  # torch.Size([43648, 8, 3])
        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).bbox for anchors_per_image in anchors], dim=0)  # torch.Size([43648, 4])
        N = len(labels)

        ###################################### Process Pred info & Extension #####################################
        # Note that GT info is in list form while the output of network is in layer form. Transform is needed.
        pred_cls_flatten, pred_reg_flatten = concat_box_prediction_layers(pred_cls, pred_reg)  # torch.Size([43648, 1]) torch.Size([43648, 16])

        ###################################### Cls Loss #####################################
        valid_cls_inds = torch.nonzero(labels_flatten >= 0).squeeze(1)
        cls_loss = self.cls_loss_func(pred_cls_flatten[valid_cls_inds], labels_flatten[valid_cls_inds])

        ###################################### 3D Loss #####################################
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)  # Fetch all valid anchors in GT. (all taged in "1")
        if pos_inds.numel() > 0:  # if any anchor is selected, count the loss of this point.
            pred_reg_flatten = pred_reg_flatten[pos_inds]               # torch.Size([78?]) start from class 0
            cls_label_flatten = labels_flatten[pos_inds] - 1            # torch.Size([78?, 16]) # start from class 0
            reg_targets_flatten = reg_targets_flatten[pos_inds]         # torch.Size([78?, 16])
            aux_raw_boxes_flatten = aux_raw_boxes_flatten[pos_inds]     # torch.Size([78?, 4])
            aux_3D_in_camera_frame_flatten = aux_3D_in_camera_frame_flatten[pos_inds]   # torch.Size([78?, 8, 3])
            anchors_flatten = anchors_flatten[pos_inds]                 # torch.Size([78?, 4])

            # all training cases in a batch
            reg_loss = self.ObjectSpaceLoss(
                pred_reg_flatten,                   # Pred: Selected prediction             torch.Size([78?, 16])
                aux_3D_in_camera_frame_flatten,     # GT: 8 3D points in camera frame       torch.Size([78?, 8, 3])
                cls_label_flatten,                  # GT: cls of selected anchors           torch.Size([78?])
                anchors_flatten                     # Anchor: size of selected anchors      torch.Size([78?, 4])
                )
        else:
            reg_loss = pred_reg_flatten.sum()

        return cls_loss * self.loss_weight_cls, reg_loss * self.loss_weight_reg
