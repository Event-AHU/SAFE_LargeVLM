import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.utils import AverageMeter, to_scalar, time_str
device = "cuda" if torch.cuda.is_available() else "cpu"

img_count=0
def batch_trainer(epoch, model,ViT_model, train_loader, criterion, optimizer):
    global img_count
    model.train()
    ViT_model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()
    #prompt_loss_meter= AverageMeter()
    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []
    save_name=[]
    save_event_name=[]
    lr = optimizer.param_groups[0]['lr']
    #lr = 1e-3
    print(f'learning rate whith VTB:{lr}')
    for step, (imgs, gt_label, imgname, label_v, event, gt_event_label, eventname) in enumerate(train_loader):
        for elem in imgname :
            save_name.append(elem)
            save_event_name.append(elem)
        img_count+=imgs.shape[0]#32
        batch_time = time.time()

        optimizer.zero_grad()

        imgs, gt_label = imgs.to(device), gt_label.to(device)
        event, gt_event_label = event.to(device), gt_event_label.to(device)
                
        output = model(imgs, event,  ViT_model=ViT_model)
        
        train_loss  = criterion(output, gt_label)
        print('==>> train_loss', train_loss)
        
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        loss_meter.update(to_scalar(train_loss))

        gt_list.append(gt_label.cpu().numpy())
        # train_probs = torch.sigmoid(output)
        preds_probs.append(output.detach().cpu().numpy())

        log_interval = 2000
        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {(time.time() - batch_time)/imgs.shape[0]:.4f}s ',
                  f'train_loss:{loss_meter.val:.4f}')
    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f},img_num:{img_count}')
    img_count=0
    return train_loss, gt_label, preds_probs

def valid_trainer(epoch,model,ViT_model, valid_loader, criterion):
    
    model.eval()
    ViT_model.eval()
    loss_meter = AverageMeter()

    preds_probs = []
    gt_list = []
    save_name=[]
    save_event_name=[]
    features = []
    with torch.no_grad():
        for step, (imgs, gt_label, imgname, label_v, event, gt_event_label, eventname) in enumerate(valid_loader):
            for elem in imgname :
                save_name.append(elem)#save_name长度=batchsize=32
            for elem in eventname :
                save_event_name.append(elem)
            imgs = imgs.cuda()
            gt_label, gt_event_label = gt_label.cuda(),gt_event_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            output = model(imgs, event,  ViT_model=ViT_model)
            breakpoint()            
            
            valid_loss = criterion(output, gt_label) 
            #print('==>> valid_loss', valid_loss)
            # valid_probs = torch.sigmoid(output)
            preds_probs.append(output.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))

    valid_loss = loss_meter.avg
    
    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return valid_loss, gt_label, preds_probs
