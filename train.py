import os
import pprint
from collections import OrderedDict, defaultdict
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from torch import nn, optim
from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics, get_signle_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler, make_scheduler

import torch.nn.functional as F

from CLIP.clip import clip
from CLIP.clip.model import *
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn.parallel
from torch.utils.data.distributed import DistributedSampler

def main(args):
    ViT_model, ViT_preprocess = clip.load("ViT-B/16", device=device,download_root='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTF_PAR-main/model') 
    ViT_model = ViT_model.float()
    # 将ViT_model移动到当前设备
    ViT_model = ViT_model.to(args.local_rank)
    
    # 初始化进程组
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)  # 设置当前设备
    log_dir = os.path.join('logs', args.dataset)
    tb_writer = SummaryWriter('/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/lidong/VTF_PAR-main/CaptionCLIP-ViT-B/tensorboardX/exp')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    select_gpus(args.gpus)

    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    
    train_tsfm, valid_tsfm = get_transform(args)

    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=train_tsfm)

    # 使用DistributedSampler
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=(train_sampler is None),
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler
    )

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

    labels = train_set.label
    sample_weight = labels.mean(0)
    
    model = TransformerClassifier(train_set.attr_num, attr_words=train_set.attributes)
    # 使用DistributedDataParallel包装模型
    # model = torch.nn.parallel.DistributedDataParallel(model.cuda())
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), find_unused_parameters=True)

    criterion = nn.CrossEntropyLoss()

    lr = args.lr
    epoch_num = args.epoch
    start_epoch = 1
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    params_to_update = list(model.parameters()) + list(ViT_model.parameters())
    optimizer = optim.AdamW(params_to_update, lr=lr, weight_decay=1e-4)
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr)

    trainer(args=args,
            epoch=epoch_num,
            model=model,
            ViT_model=ViT_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            path=log_dir,
            tb_writer=tb_writer,
            start_epoch=start_epoch,
            save_interval=args.save_interval)


def trainer(args, epoch, model, ViT_model, train_loader, valid_loader, criterion, optimizer, scheduler, path,
            tb_writer, start_epoch, save_interval):
    max_ma, max_acc, max_f1 = 0, 0, 0
    start = time.time()
    best_accuracy = 0
    for i in range(start_epoch, epoch + 1):
        scheduler.step(1)#固定lr
        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            ViT_model=ViT_model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer
        )

        valid_loss, valid_gt, valid_probs = valid_trainer(
            epoch=epoch,
            model=model,
            ViT_model=ViT_model,
            valid_loader=valid_loader,
            criterion=criterion,
        )

        if args.dataset == 'poker':
            train_preds = train_probs.argmax(axis=1)
            train_gt = train_gt.argmax(axis=1)
            correct_predictions = (train_preds == train_gt).sum()
            train_accuracy = correct_predictions / len(train_gt)
            print('===>>train_accuracy = ', train_accuracy)

            valid_preds = valid_probs.argmax(axis=1)
            valid_gt = valid_gt.argmax(axis=1)
            valid_correct_predictions = (valid_preds == valid_gt).sum()
            valid_accuracy = valid_correct_predictions / len(valid_gt)
            print('===>>valid_accuracy = ', valid_accuracy)

        # Save checkpoint
        if i % save_interval == 0:
            ckpt_path = os.path.join(path, f'checkpoint_{i}.pth')
            save_checkpoint(model, ViT_model, optimizer, epoch=i, path=ckpt_path)
        
        # 如果当前精度高于最高精度值，则保存当前checkpoint
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_ckpt_path = os.path.join(path, 'best_checkpoint.pth')
            save_checkpoint(model, ViT_model, optimizer, epoch=i, path=best_ckpt_path)

    end = time.time()
    elapsed = end - start
    print('===>>Elapsed Time: [%.2f h %.2f m %.2f s]' % (elapsed // 3600, (elapsed % 3600) // 60, (elapsed % 60)))

    print('===>>Training Complete...')


def save_checkpoint(model,  ViT_model, optimizer, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'ViT_model_state_dict': ViT_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)


if __name__ == '__main__':
    parser = argument_parser()
    parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving checkpoints')
    parser.add_argument('--local_rank', type=int, help='Local rank for distributed training')
    args = parser.parse_args()
    main(args)