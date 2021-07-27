# -*- coding: utf-8 -*-

import time
from tqdm import tqdm
import torch
from evaluator import *

# Training
def train(args, model, device, train_loader, optimizer, epoch, criterions, part=None):
    # statistics recoder
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1 = AverageMeter()

    # start training over batch
    model.train()
    end = time.time()
    progress_bar = tqdm(train_loader, ncols=120)
    for i, (data, targets) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        # in case of simultaneous learning, targets = [class, PH]
        data_time.update(time.time() - end)
        data = data.to(device,non_blocking=True)

        outputs = model(data)
        if isinstance(part, list): # in case of simultaneous learning, criterion[0] is for classification, [1] is for regression
            output2 = outputs[:,slice(part[1][0],part[1][1])]
            output1 = outputs[:,slice(part[0][0],part[0][1])]
            target2 = targets[1].to(device, non_blocking=True)
            target1 = targets[0].to(device, non_blocking=True)
            #print(output.shape,output2.shape,target.shape,target2.shape, part)
            loss2 = criterions[1](output2, target2)
            losses2.update(loss2.item(), data.size(0))
        else:
            target1 = targets.to(device)
            if part is not None:
                output1 = outputs[:,slice(*part)]
            else:
                output1 = outputs
            loss2 = 0
            losses2.update(0.0, data.size(0))
        
        loss1 = criterions[0](output1, target1)
        if target1.dtype == torch.float32:
            acc1 = [0]
        else:
            acc1 = accuracy(output1, target1, topk=(1,))[0]

        losses1.update(loss1.item(), data.size(0))
        top1.update(acc1[0], data.size(0))

        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if (i+1) % args.log_interval == 0 or (i+1) == len(train_loader):
        progress_bar.set_postfix(
            #'epoch: [{0}][{1}/{2}]\t'
            #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            Loss1='{:.4f}'.format(losses1.avg),
            Loss2='{:.4f}'.format(losses2.avg),
            Acc1='{:.4f}'.format(top1.avg))
#             'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
#             epoch, i+1, len(train_loader), 
#             batch_time=batch_time,
#             data_time=data_time, loss1=losses1, loss2=losses2, top1=top1))
    
    return(losses1.avg,top1.avg,losses2.avg)

# Validation
def validate(args, model, device, val_loader, criterions, part=None):
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (data, targets) in enumerate(val_loader):
            data = data.to(device,non_blocking=True)
            outputs = model(data)
            
            if isinstance(part, list): # simultaneous
                output2 = outputs[:,slice(*part[1])]
                output1 = outputs[:,slice(*part[0])]
                target2 = targets[1].to(device,non_blocking=True)
                target1 = targets[0].to(device,non_blocking=True)
                loss2 = criterions[1](output2, target2)
                losses2.update(loss2.item(), data.size(0))
            else:
                if part is not None:
                    output1 = outputs[:,slice(*part)]
                else:
                    output1 = outputs
                target1 = targets.to(device)
                loss2 = 0
                losses2.update(0.0, data.size(0))

            loss1 = criterions[0](output1, target1)

            k = min(args.numof_classes,5)
            if target1.dtype == torch.float32:
                acc1, acc5 = [0], [0]
            else:
                acc1, acc5 = accuracy(output1, target1, topk=(1,k))
            losses1.update(loss1.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))
        
    tqdm.write('Test: Loss1 ({loss1.avg:.4f})\t'
            'Test: Loss2 ({loss2.avg:.4f})\t'
            'Acc@1 ({top1.avg:.3f})\t'
            'Acc@{k} ({top5.avg:.3f})'.format(
            loss1=losses1, loss2=losses2, k=k, top1=top1, top5=top5))
    return losses1.avg, top1.avg, losses2.avg

