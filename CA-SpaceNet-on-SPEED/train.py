from email.mime import image
from email.policy import strict
from nis import maps
import os
import sys
import time
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import cv2
import math

from backbone import darknet53
from core.argument import get_args
from core.dataset import BOP_Dataset_train, BOP_Dataset_test, collate_fn_train, collate_fn_test
from core.model import PoseModule, PoseModule_Casual, Counterfactual_path
from core.scheduler import WarmupScheduler
import core.transform as transform
from core.evaluate import evaluate
from core.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    DistributedSampler,
    all_gather,
)
from core.utils import (
    load_bop_meshes,
    visualize_pred,
    print_accuracy_per_class,
)
from tensorboardX import SummaryWriter

# reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
np.random.seed(0)

# close shared memory of pytorch
if True:
    # https://github.com/huaweicloud/dls-example/issues/26
    from torch.utils.data import dataloader
    from torch.multiprocessing import reductions
    from multiprocessing.reduction import ForkingPickler
    default_collate_func = dataloader.default_collate
    def default_collate_override(batch):
        dataloader._use_shared_memory = False
        return default_collate_func(batch)
    setattr(dataloader, 'default_collate', default_collate_override)
    for t in torch._storage_classes:
        if sys.version_info[0] == 2:
            if t in ForkingPickler.dispatch:
                del ForkingPickler.dispatch[t]
        else:
            if t in ForkingPickler._extra_reducers:
                del ForkingPickler._extra_reducers[t]

def accumulate_dicts(data):
    all_data = all_gather(data)

    if get_rank() != 0:
        return

    data = {}

    for d in all_data:
        data.update(d)

    return data

@torch.no_grad()
def valid(cfg, epoch, loader, model, device, logger=None):
    torch.cuda.empty_cache()

    model.eval()

    if get_rank() == 0:
        pbar = tqdm(enumerate(loader), total=len(loader), ncols=120)
    else:
        pbar = enumerate(loader)

    preds = {}

    for idx, (images, targets, meta_infos) in pbar:
        model.zero_grad()

        images = images.to(device)  # [8, 3, 960, 960]
        targets = [target.to(device) for target in targets]

        # from nni.compression.pytorch.utils.counter import count_flops_params
        # print(images.tensors.shape)
        # flops, params, results = count_flops_params(model, images.tensors)
        # print(flops)
        # print(results)
        # exit(0)
        
        pred, xy2d_nps = model(images,[],targets=targets)

        # if get_rank() == 0:  # and idx % 10 == 0
        bIdx = 0
        imgpath, imgname = os.path.split(meta_infos[bIdx]['path'])
        name_prefix = imgpath.replace(os.sep, '_').replace('.', '') + '_' + os.path.splitext(imgname)[0]

        rawImg, visImg, gtImg = visualize_pred(
            images.tensors[bIdx], 
            targets[bIdx], 
            pred[bIdx], 
            cfg['INPUT']['PIXEL_MEAN'], 
            cfg['INPUT']['PIXEL_STD']
            )
        # cv2.imwrite(cfg['RUNTIME']['WORKING_DIR'] + name_prefix + '.png', rawImg)
        # cv2.imwrite(cfg['RUNTIME']['WORKING_DIR'] + name_prefix + '_pred.png', visImg)
        # cv2.imwrite(cfg['RUNTIME']['WORKING_DIR'] + name_prefix + '_gt.png', gtImg)

        # 绘制预测点
        for i in range(len(xy2d_nps)):
            for j in range(len(xy2d_nps[i])):
                pt = (int(xy2d_nps[i][j][0]), int(xy2d_nps[i][j][1]))
                if j < 4:
                    gtImg = cv2.circle(gtImg, pt, 1, [512, 0, 0], 0)
                else:
                    gtImg = cv2.circle(gtImg, pt, 1, [0, 0, 512], 0)
        
        # cv2.imwrite(cfg['RUNTIME']['WORKING_DIR'] + name_prefix + '_points.png', gtImg)

        # pred = [p.to('cpu') for p in pred]
        for m, p in zip(meta_infos, pred):
            preds.update({m['path']:{
                'meta': m,
                'pred': p
            }})

    preds = accumulate_dicts(preds)

    if get_rank() != 0:
        return
     
    # evaluate:
    # key: ./data/SPEED/images/train/img008760.jpg
    S_R_mean = []
    S_t_mean = []
    S_total_mean = []
    cnt = 0
    for key_ in preds.keys():
        # GT:
        GT_R = preds[key_]['meta']['rotations'][0]  # R:3*3 
        GT_t = preds[key_]['meta']['translations'][0]  # T:3*1 array([[-0.21081 ], [-0.094466], [ 6.705986]])
        # Pred:
        if len(preds[key_]['pred']) == 0:
            continue
        Pred_R = preds[key_]['pred'][0][2]  # R:3*3 
        Pred_t = preds[key_]['pred'][0][3]  # T:3*1 array([[-0.21546072], [-0.10613697], [ 6.65623294]])

        from pyquaternion import Quaternion
        import math
        GT_q = np.array(list(Quaternion(matrix=GT_R)))
        Pred_q = np.array(list(Quaternion(matrix=Pred_R)))

        # Print
        # print('GT_q:', GT_q, '\nPred_q:', Pred_q)
        # print('GT_t:', GT_t.reshape(-1), '\nPred_t:', Pred_t.reshape(-1))
        # print()

        S_R = 2*math.acos(min(1, abs(np.dot(GT_q, -1 * Pred_q))))
        S_t = np.linalg.norm(GT_t - Pred_t)/np.linalg.norm(GT_t)
        if S_t > 1000:  # if S_t is too big, jump it
            continue
        S_total = S_R + S_t

        S_R_mean.append(S_R)
        S_t_mean.append(S_t)
        S_total_mean.append(S_total)

        cnt = cnt + 1

    S_R_mean = np.array(S_R_mean)
    S_t_mean = np.array(S_t_mean)
    S_total_mean = np.array(S_total_mean)

    # for i in range(len(S_R_mean)):
    #     # Print
    #     print('Err_q:', S_R_mean[i], 'Err_t:', S_t_mean[i], 'Err_all:', S_total_mean[i])

    print('S_R_m:%.6f    S_t_m:%.6f    S_total_m:%.6f    Test:(%d/2000)'%(np.mean(S_R_mean), np.mean(S_t_mean), np.mean(S_total_mean), cnt))

    logger.add_scalars('S_R_mean', {'R':np.mean(S_R_mean)}, epoch)
    logger.add_scalars('S_t_mean', {'t':np.mean(S_t_mean)}, epoch)
    logger.add_scalars('S_total_mean', {'total':np.mean(S_total_mean)}, epoch)

def train(cfg, epoch, max_epoch, loader, model, model_c, optimizer, scheduler, device, logger=None):
    model.train()
    model_c.eval()  # 反事实模型不用训练

    if get_rank() == 0:
        pbar = tqdm(enumerate(loader), total=len(loader), ncols=120)
    else:
        pbar = enumerate(loader)

    
    for idx, (images, masked, targets, _) in pbar:
        model.zero_grad()
        
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        # 反事实图像准备
        images_c = masked.to(device)

        # 反事实图像的处理
        features_c = model_c(images_c)

        _, loss_dict = model(images, features_c, targets=targets)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_reg = loss_dict['loss_reg'].mean()
        loss_sim = loss_dict['loss_sim'].mean()

        loss = loss_cls + loss_reg + loss_sim
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        loss_reduced = reduce_loss_dict(loss_dict)
        loss_cls = loss_reduced['loss_cls'].mean().item()
        loss_reg = loss_reduced['loss_reg'].mean().item()
        loss_sim = loss_reduced['loss_sim'].mean().item()

        if get_rank() == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar_str = (("epoch: %d/%d, lr:%.6f, cls:%.4f, reg:%.4f, sim:%.4f") % (epoch+1, max_epoch, current_lr, loss_cls, loss_reg, loss_sim))
            pbar.set_description(pbar_str)

            # writing log to tensorboard
            if logger and idx % 10 == 0:
                # totalStep = (epoch * len(loader) + idx) * args.batch * args.n_gpu
                totalStep = (epoch * len(loader) + idx) * cfg['SOLVER']['IMS_PER_BATCH']
                logger.add_scalar('training/learning_rate', current_lr, totalStep)
                logger.add_scalar('training/loss_cls', loss_cls, totalStep)
                logger.add_scalar('training/loss_reg', loss_reg, totalStep)
                logger.add_scalar('training/loss_sim', loss_sim, totalStep)
                logger.add_scalar('training/loss_all', (loss_cls + loss_reg), totalStep)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return sampler.RandomSampler(dataset)

    else:
        return sampler.SequentialSampler(dataset)

def dataset_create(cfg):
    # device = 'cuda'
    device = cfg['RUNTIME']['RUNNING_DEVICE']

    internal_K = np.array(cfg['INPUT']['INTERNAL_K']).reshape(3,3)

    train_trans = transform.Compose(
        [
            transform.Resize(
                cfg['INPUT']['INTERNAL_WIDTH'], 
                cfg['INPUT']['INTERNAL_HEIGHT'], internal_K),
            transform.RandomShiftScaleRotate(
                cfg['SOLVER']['AUGMENTATION_SHIFT'],  # 0.1
                cfg['SOLVER']['AUGMENTATION_SCALE'],  # 0.1
                cfg['SOLVER']['AUGMENTATION_ROTATION'],  # 180
                cfg['INPUT']['INTERNAL_WIDTH'],  # 512
                cfg['INPUT']['INTERNAL_HEIGHT'],  # 512
                internal_K),
            transform.Normalize(
                cfg['INPUT']['PIXEL_MEAN'],  # [0.485, 0.456, 0.406]
                cfg['INPUT']['PIXEL_STD']),  # [0.229, 0.224, 0.225]
            transform.ToTensor(),
        ]
    )

    valid_trans = transform.Compose(
        [
            transform.Resize(
                cfg['INPUT']['INTERNAL_WIDTH'], 
                cfg['INPUT']['INTERNAL_HEIGHT'], 
                internal_K),
            transform.Normalize(
                cfg['INPUT']['PIXEL_MEAN'], 
                cfg['INPUT']['PIXEL_STD']),
            transform.ToTensor(), 
        ]
    )

    train_set = BOP_Dataset_train(
        cfg['DATASETS']['TRAIN'], 
        cfg['DATASETS']['BBOX_FILE'], 
        cfg['DATASETS']['TRAIN_ANN'], 
        train_trans,
        cfg['SOLVER']['STEPS_PER_EPOCH'] * cfg['SOLVER']['IMS_PER_BATCH'],
        training = True)
    valid_set = BOP_Dataset_test(
        cfg['DATASETS']['VALID'],
        cfg['DATASETS']['BBOX_FILE'], 
        cfg['DATASETS']['VALID_ANN'], 
        valid_trans,
        training = False)

    return device, train_set, valid_set

def dataloader_create(cfg, train_set, valid_set):
    batch_size_per_gpu = int(cfg['SOLVER']['IMS_PER_BATCH'] / cfg['RUNTIME']['N_GPU'])
    max_epoch = math.ceil(cfg['SOLVER']['MAX_ITER'] * cfg['SOLVER']['IMS_PER_BATCH'] / len(train_set))
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size_per_gpu,
        sampler=data_sampler(train_set, shuffle=True, distributed=cfg['RUNTIME']['DISTRIBUTED']),
        num_workers=cfg['RUNTIME']['NUM_WORKERS'],
        collate_fn=collate_fn_train(cfg['INPUT']['SIZE_DIVISIBLE']),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        sampler=data_sampler(valid_set, shuffle=False, distributed=cfg['RUNTIME']['DISTRIBUTED']),
        num_workers=cfg['RUNTIME']['NUM_WORKERS'],
        collate_fn=collate_fn_test(cfg['INPUT']['SIZE_DIVISIBLE']),
    )

    return max_epoch, train_loader, valid_loader

def model_create(cfg):
    start_epoch = 0

    if cfg['MODEL']['BACKBONE'] == 'darknet53':
        backbone = darknet53(pretrained=True)
        backbone_c = darknet53(pretrained=True)
    else:
        print("unsupported backbone!")
        assert(0)
    
    # Instantiate two models
    model = PoseModule_Casual(cfg, backbone)
    model_c = Counterfactual_path(cfg, backbone_c)
    model, model_c = model.to(device), model_c.to(device)

    # Compute model size
    total_params_count = sum(p.numel() for p in model.parameters())
    print("Model size of the PoseModule_Casual model: %d parameters" % total_params_count)
    total_params_count = sum(p.numel() for p in model_c.parameters())
    print("Model size of the Counterfactual Factual Path: %d parameters" % total_params_count)

    return start_epoch, model, model_c

def optimizer_create(cfg):
    base_lr = cfg['SOLVER']['BASE_LR'] / cfg['RUNTIME']['N_GPU']
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),  # 目前只放了主网络的两个FPN和一个Head
        lr = 0, # the learning rate will be taken care by scheduler
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )

    scheduler_batch = WarmupScheduler(
        optimizer, base_lr, 
        cfg['SOLVER']['MAX_ITER'], 
        cfg['SOLVER']['SCHEDULER_POLICY'], 
        cfg['SOLVER']['SCHEDULER_PARAMS']
        )

    return optimizer, scheduler_batch

def load_pretraind_weights(cfg, model, model_c):
    # Load pre-trained weights
    if os.path.exists(cfg['RUNTIME']['WEIGHT_FILE']):  # --weight_file
        # try:
        chkpt = torch.load(cfg['RUNTIME']['WEIGHT_FILE'], map_location='cpu')  # load checkpoint
        if 'model' in chkpt:
            exit(0)
            assert('steps' in chkpt and 'optim' in chkpt)
            # scheduler_batch.step_multiple(chkpt['steps'])
            # start_epoch = int(chkpt['steps'] * cfg['SOLVER']['IMS_PER_BATCH'] / len(train_set))
            # model.load_state_dict(chkpt['model'])
            # optimizer.load_state_dict(chkpt['optim'])
            # update working dir
            # cfg['RUNTIME']['WORKING_DIR'] = os.path.split(cfg['RUNTIME']['WEIGHT_FILE'])[0] + '/'
            print('----- Weights and optimzer are loaded from ' + cfg['RUNTIME']['WEIGHT_FILE'])
        else:
            # for i in list(chkpt.keys()):
            #     if 'fpn' in i:
            #         print(i)
            #         value = chkpt[i]
            #         x = i.replace("fpn", "fpn_sub")
            #         chkpt.update({x:value})
            
            model.load_state_dict(chkpt, strict=True)  # 忽略不匹配层
            model_c.load_state_dict(chkpt, strict=False)  # 忽略不匹配层
            print('+++++ Weights from are loaded from ' + cfg['RUNTIME']['WEIGHT_FILE'])
        # except:
        #     pass
    else:
        pass

    return model, model_c

def working_dir_create(cfg):
    # load weight and create working_dir dynamically
    timestr = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
    name_wo_ext = os.path.splitext(os.path.split(cfg['RUNTIME']['CONFIG_FILE'])[1])[0]
    working_dir = 'working_dirs' + '/' + name_wo_ext + '/' + timestr + '/'
    cfg['RUNTIME']['WORKING_DIR'] = working_dir

    print("working directory: " + cfg['RUNTIME']['WORKING_DIR'])
    if get_rank() == 0:
        os.makedirs(cfg['RUNTIME']['WORKING_DIR'], exist_ok=True)
        logger = SummaryWriter(cfg['RUNTIME']['WORKING_DIR'])

    # write cfg to working_dir
    with open(cfg['RUNTIME']['WORKING_DIR'] + 'cfg.json', 'w') as f:
        json.dump(cfg, f, indent=4, sort_keys=True)
    
    return logger, cfg

if __name__ == '__main__':
    cfg = get_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    cfg['RUNTIME']['N_GPU'] = n_gpu
    cfg['RUNTIME']['DISTRIBUTED'] = n_gpu > 1

    '''
    if cfg['RUNTIME']['DISTRIBUTED']:
        torch.cuda.set_device(cfg['RUNTIME']['LOCAL_RANK'])
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        synchronize()
    '''
    #### Create datasets
    device, train_set, valid_set = dataset_create(cfg)

    #### Create dataloader
    max_epoch, train_loader, valid_loader = dataloader_create(cfg, train_set, valid_set)

    #### Create model
    start_epoch, model, model_c = model_create(cfg)

    #### Load pre-trained weights and optim
    model, model_c = load_pretraind_weights(cfg, model, model_c)

    #### Create working_dir dynamically
    for i, p in enumerate(model.backbone.parameters()):
        p.requires_grad = False
    for i, p in enumerate(model_c.parameters()):
        p.requires_grad = False

    #### Create optimizer
    optimizer, scheduler_batch = optimizer_create(cfg)

    #### Create working_dir dynamically
    logger, cfg = working_dir_create(cfg)

    '''
    if cfg['RUNTIME']['DISTRIBUTED']:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg['RUNTIME']['LOCAL_RANK']],
            output_device=cfg['RUNTIME']['LOCAL_RANK'],
            broadcast_buffers=False,
        )
        model = model.module
    '''
    
    #### Training
    for epoch in range(start_epoch, max_epoch):
        # train(cfg, epoch, max_epoch, train_loader, model, model_c, optimizer, scheduler_batch, device, logger=logger)

        valid(cfg, epoch, valid_loader, model, device, logger=logger)

        if get_rank() == 0:
            torch.save({
                'steps': (epoch + 1) * int(len(train_set) / cfg['SOLVER']['IMS_PER_BATCH']), 
                'model': model.state_dict(), 
                'optim': optimizer.state_dict(),
                },
                cfg['RUNTIME']['WORKING_DIR'] + 'latest.pth',
            )
            if epoch > (max_epoch - 5):  # 从25之后，每个epoch都保存
                torch.save(model.state_dict(), cfg['RUNTIME']['WORKING_DIR'] + '%d.pth'% (epoch + 1))

            if epoch == (max_epoch - 1):
                torch.save(model.state_dict(), cfg['RUNTIME']['WORKING_DIR'] + 'final.pth')

    # output final info
    if get_rank() == 0:
        timestr = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
        commandstr = ' '.join([str(elem) for elem in sys.argv]) 
        final_msg = ("finished at: %s\nworking_dir: %s\ncommands:%s" % (timestr, cfg['RUNTIME']['WORKING_DIR'], commandstr))
        with open(cfg['RUNTIME']['WORKING_DIR'] + 'info.txt', 'w') as f:
            f.write(final_msg)
        print(final_msg)


# 30:
# S_R_m:0.122199   S_t_m:0.040064          S_total_m:0.162263