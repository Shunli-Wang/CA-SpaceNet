import os
import sys
import time
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

import numpy as np
import cv2

from backbone import darknet53
from core.argument import get_args
from core.dataset import BOP_Dataset_test, collate_fn_test
from core.model import PoseModule_Casual
import core.transform as transform
from core.evaluate import evaluate
from core.utils import visualize_accuracy_per_depth
from core.distributed import (
    get_rank,
    synchronize,
)
from train import (
    valid,
    data_sampler,
)


from tensorboardX import SummaryWriter

torch.manual_seed(0)
np.random.seed(0)

def working_dir_create(cfg):
    # create working_dir dynamically
    timestr = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
    name_wo_ext = os.path.splitext(os.path.split(cfg['RUNTIME']['CONFIG_FILE'])[1])[0]
    working_dir = 'working_dirs' + '/' + name_wo_ext + '/test_' + timestr + '/'
    cfg['RUNTIME']['WORKING_DIR'] = working_dir
    print("working directory: " + cfg['RUNTIME']['WORKING_DIR'])

    if get_rank() == 0:
        os.makedirs(cfg['RUNTIME']['WORKING_DIR'], exist_ok=True)
        logger = SummaryWriter(cfg['RUNTIME']['WORKING_DIR'])

    return logger, cfg

def dataset_create(cfg):
    # device = 'cuda'
    device = cfg['RUNTIME']['RUNNING_DEVICE']

    internal_K = np.array(cfg['INPUT']['INTERNAL_K']).reshape(3,3)

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

    valid_set = BOP_Dataset_test(
        cfg['DATASETS']['TEST'],
        cfg['DATASETS']['MESH_DIR'], 
        cfg['DATASETS']['BBOX_FILE'], 
        valid_trans,
        training = False)

    return device, valid_set

def model_create(cfg):
    if cfg['MODEL']['BACKBONE'] == 'darknet53':
        backbone = darknet53(pretrained=False)
    else:
        print("unsupported backbone!")
        assert(0)
        
    model = PoseModule_Casual(cfg, backbone)
    model = model.to(device)

    return model

def load_pretraind_weights(cfg, model):
    # load weight
    if os.path.exists(cfg['RUNTIME']['WEIGHT_FILE']):
        try:
            chkpt = torch.load(cfg['RUNTIME']['WEIGHT_FILE'], map_location='cpu')  # load checkpoint
            if 'model' in chkpt:
                chkpt = chkpt['model']
            # model.load_state_dict(chkpt) # strict
            # loose loading
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in chkpt.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            # 
            print('Weights are loaded from ' + cfg['RUNTIME']['WEIGHT_FILE'])
        except:
            print('Loading weights from %s is failed' % (cfg['RUNTIME']['WEIGHT_FILE']))
            print("Random initialized weights.")
    else:
        print("Random initialized weights.")

    return model

def dataloader_create(cfg, valid_set):
    batch_size_per_gpu = int(cfg['TEST']['IMS_PER_BATCH'] / cfg['RUNTIME']['N_GPU'])
    if batch_size_per_gpu == 0:
        print('ERROR: %d GPUs for %d batch(es)' % (cfg['RUNTIME']['N_GPU'], cfg['TEST']['IMS_PER_BATCH']))
        assert(0)

    valid_loader = DataLoader(
        valid_set,
        batch_size=1,  # 
        sampler=data_sampler(valid_set, shuffle=False, distributed=cfg['RUNTIME']['DISTRIBUTED']),
        num_workers=cfg['RUNTIME']['NUM_WORKERS'],
        collate_fn=collate_fn_test(cfg['INPUT']['SIZE_DIVISIBLE']),
    )
    return valid_loader

if __name__ == '__main__':

    cfg = get_args()

    #### Create working_dir dynamically
    logger, cfg = working_dir_create(cfg)

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    cfg['RUNTIME']['N_GPU'] = n_gpu
    cfg['RUNTIME']['DISTRIBUTED'] = n_gpu > 1
    '''
    # if cfg['RUNTIME']['DISTRIBUTED']:
    #     torch.cuda.set_device(cfg['RUNTIME']['LOCAL_RANK'])
    #     torch.distributed.init_process_group(backend='gloo', init_method='env://')
    #     synchronize()
    '''
    
    #### Create datasets
    device, valid_set = dataset_create(cfg)

    #### Create model
    model = model_create(cfg)

    #### Load pre-trained weights
    model = load_pretraind_weights(cfg, model)

    #### Create dataloader
    valid_loader = dataloader_create(cfg, valid_set)

    '''
    # if cfg['RUNTIME']['DISTRIBUTED']:
    #     model = nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[cfg['RUNTIME']['LOCAL_RANK']],
    #         output_device=cfg['RUNTIME']['LOCAL_RANK'],
    #         broadcast_buffers=False,
    #     )
    #     model = model.module
    '''

    accuracy_adi_per_class, \
    accuracy_rep_per_class, \
    accuracy_adi_per_depth, \
    accuracy_rep_per_depth, \
    depth_range = valid(cfg, 0, valid_loader, model, device, logger=logger)

    visImg = visualize_accuracy_per_depth(
        accuracy_adi_per_class, 
        accuracy_rep_per_class, 
        accuracy_adi_per_depth, 
        accuracy_rep_per_depth, 
        depth_range
    )

    visFileName = cfg['RUNTIME']['WORKING_DIR'] + 'error_statistics_per_depth.png'
    cv2.imwrite(visFileName, visImg)
    print("Error statistics for each depth bin are saved to '%s'" % visFileName)
