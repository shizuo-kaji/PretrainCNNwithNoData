# -*- coding: utf-8 -*-

import time
import torch
from evaluator import *

# Training
def train(args, model, device, train_loader, optimizer, epoch, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        if args.persistence is not None:
            acc1 = 0
        else:
            acc1 = accuracy(output, target, topk=(1,))[0][0]

        losses.update(loss.item()/data.size(0), data.size(0))
        top1.update(acc1, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.log_interval == 0:
            print('epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), 
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            
    return(losses.avg,top1.avg)

# Validation
def validate(args, model, device, val_loader, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)

            k = min(args.numof_classes,5)
            if args.persistence is not None:
                acc1, acc5 = [0], [0]
            else:
                acc1, acc5 = accuracy(output, target, topk=(1,k))
            losses.update(loss.item()/data.size(0), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))
        
        print('Test: Loss ({loss.avg:.4f})\t'
              'Acc@1 ({top1.avg:.3f})\t'
              'Acc@{k} ({top5.avg:.3f})'.format(
               loss=losses, k=k, top1=top1, top5=top5))
    return losses.avg, top1.avg

