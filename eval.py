import os
import pprint
from collections import OrderedDict, defaultdict
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from torch import nn, optim
from batch_engine import valid_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics, get_signle_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus

import torch.nn.functional as F

from CLIP.clip import clip
from CLIP.clip.model import *
from tensorboardX import SummaryWriter

import torch.distributed as dist
import torch.nn.parallel
from torch.utils.data.distributed import DistributedSampler

def main(args):
    ViT_model, ViT_preprocess = clip.load("ViT-B/16", device=device,download_root='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTF_PAR-main/model') 
    ViT_model = ViT_model.float()
    ViT_model = ViT_model.to(args.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    valid_tsfm = get_transform(args)[1]

    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)

    valid_sampler = DistributedSampler(valid_set)
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=valid_sampler
    )

    model = TransformerClassifier(valid_set.attr_num, attr_words=valid_set.attributes)
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), find_unused_parameters=True)
    
    checkpoint = torch.load('/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTF_PAR-main/logs/poker/best_checkpoint.pth',map_location='cuda:0')
    #可视化
    #checkpoint = torch.load('/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTF_PAR-main/Visualization/VTF/best_checkpoint.pth',map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    ViT_model.load_state_dict(checkpoint['ViT_model_state_dict'], strict=False)

    criterion = nn.CrossEntropyLoss()
        
    valid_loss, valid_gt, valid_probs = valid_trainer(
        epoch=1,
        model=model,
        ViT_model=ViT_model,
        valid_loader=valid_loader,
        criterion=criterion,
    )

    if args.dataset == 'poker':
        valid_preds = valid_probs.argmax(axis=1)
        valid_gt = valid_gt.argmax(axis=1)
        valid_correct_predictions = (valid_preds == valid_gt).sum()
        valid_accuracy = valid_correct_predictions / len(valid_gt)
        print('===>>valid_accuracy = ', valid_accuracy)

    print('===>>Testing Complete...')

        
if __name__ == '__main__':
    parser = argument_parser()
    parser.add_argument('--local_rank', type=int, help='Local rank for distributed training')
    args = parser.parse_args()
    main(args)
    

