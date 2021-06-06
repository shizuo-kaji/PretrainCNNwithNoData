# -*- coding: utf-8 -*-
import os
import random
import time
import json
import numpy as np
from PIL import Image
from datetime import datetime as dt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import DatasetFolderPH
from arguments import arguments
from train_val import train, validate
from model_select import model_select
from torch.utils.tensorboard import SummaryWriter

args = arguments()

def worker_init_fn(worker_id):
    random.seed(worker_id+args.seed)

def dpp_train(rank, world_size, args):
    # setup parallelisation environment
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # create default process group
    if args.pidf == "nccl":
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        dist.init_process_group("nccl", rank=rank, world_size=world_size) # , init_method=pid)
        torch.cuda.set_device(rank)
        print("paralell trainer based on nccl")
    elif args.pidf == "gloo":
        pidf = os.path.join(args.output,"temp") ## check write permission!
        if os.path.isfile(pidf):
            os.remove(pidf)
        pid = "file:///{}".format(pidf)  
        dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=pid)
        torch.cuda.set_device(rank)
        print("paralell trainer based on gloo")

    # training dataset
    #normalize = transforms.Normalize(mean=[0.2, 0.2, 0.2], std=[0.5, 0.5, 0.5])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tr = [transforms.RandomCrop((args.crop_size,args.crop_size)),transforms.ToTensor(),normalize]
    if args.img_size > args.crop_size:
        tr = [transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BILINEAR)] + tr
    if args.affine:
        tr = [transforms.RandomAffine(degrees=(-180,180), scale=(0.5,2), shear=(-100,100,-100,100))] + tr
    train_transform = transforms.Compose(tr)
    if args.label_type == "class":
        train_dataset = datasets.ImageFolder(args.train, transform=train_transform)
        criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    else:
        PHdir = None
        if os.path.isdir(args.path2PHdir):
            print("PH will be loaded from: ",args.path2PHdir)
            PHdir = args.path2PHdir
        else:
            print("PH histogram computed on the fly")
        if args.train is None:
            print("training images are generated on the fly.")
            train_dataset = DatasetFolderPH(root=None, transform=train_transform, args=args)
        else:
            train_dataset = DatasetFolderPH(args.train, transform=train_transform, args=args)        
        criterion = nn.MSELoss(reduction='mean').to(device)

    for i in range(args.output_training_images):
        img = (train_dataset[i][0]).numpy()
        img = (255*(img-img.min())/np.ptp(img)).astype(np.uint8).transpose(1,2,0)
        Image.fromarray(img).save(os.path.join(args.output,"{:0>5}.jpg".format(i)))

    if args.pidf in ["nccl","gloo"]:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=world_size,rank = rank)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                    num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn, sampler = train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    print(len(train_dataset), "train images loaded.")


    # validation dataset
    if args.val is not None and rank==0:
        val_transform = transforms.Compose([transforms.Resize((args.crop_size,args.crop_size), interpolation=transforms.InterpolationMode.BILINEAR),
                                         transforms.ToTensor(), normalize])
        if args.label_type == "class":
            val_dataset = datasets.ImageFolder(args.val, transform=val_transform)
        else:
            val_dataset = DatasetFolderPH(args.val, transform=val_transform,args=args)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        print(len(val_dataset), "validation images loaded.")

    # setup model and optimiser
    model = model_select(args)
    if args.world_size != 1:
        if args.pidf in ["nccl","gloo"]:
            model = DDP(model.to(rank), device_ids=[rank],output_device=rank)
        else:
            model = nn.DataParallel(model)
            model = model.to(device)
    else:
        model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)
    if "sgd" in args.optimizer:
        optimizer = optim.SGD(model.parameters(), lr=args.lr_pre, momentum=args.momentum, weight_decay=args.weight_decay)
    elif "adam" in args.optimizer:
        optimizer = optim.Adam(model.parameters(), lr=args.lr_pre)
        print("using Adam.")
    if "cos" in args.optimizer:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        print("using cosine annealing.")
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//3,2*args.epochs//3], gamma=0.1)

    # checkpointing
    if args.resume:
        if args.pidf in ["nccl","gloo"]:
            dist.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(args.resume, map_location=map_location)
        else:
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("checkpoint loaded: {} (epoch {})".format(args.resume, checkpoint['epoch']))
    
    # logger
    if rank==0:
        writer = SummaryWriter(log_dir=args.logdir)
    else:
        writer = None

    # Training and Validation
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.pidf in ["nccl","gloo"]:
            train_sampler.set_epoch(epoch)
            loss, acc = train(args, model, rank, train_loader, optimizer, epoch, criterion)
        else:
            loss, acc = train(args, model, device, train_loader, optimizer, epoch, criterion)
        scheduler.step()
        if rank==0:
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Acc/train", acc, epoch)

        if args.val is not None and rank==0:
            validation_loss, val_acc  = validate(args, model, device, val_loader, criterion)
            writer.add_scalar("Loss/val", validation_loss, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)
        if ((args.save_interval>0 and epoch % args.save_interval == 0) or epoch == args.epochs) and rank==0:
            print("saving checkpoint...")
            if args.world_size>1:
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            # Save checkpoint
            saved_weight = "{}/{}_epoch{}.pth.tar".format(args.output, args.usenet, epoch)
            torch.save(model_state, saved_weight.replace('.tar',''))
            checkpoint = "{}/checkpoint_epoch{}.pth.tar".format(args.output, epoch)
            torch.save({'epoch': epoch + 1,
                        'state_dict': model_state,
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),}, checkpoint)

    if args.pidf in ["nccl","gloo"]:
        dist.destroy_process_group()

if __name__== "__main__":
    starttime = time.time()
    print(args)
    grd ="grad_" if args.gradient else ""
    dtstr = dt.now().strftime('%Y_%m%d_%H%M_{}{}_ml{}_n{}'.format(grd,args.label_type,args.max_life,args.numof_classes))
    args.logdir = os.path.join(os.path.dirname(__file__),"runs/pt/{}".format(dtstr))
    args.output = os.path.join(os.path.expanduser(args.output), dtstr)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.output, "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
    with open(os.path.join(args.logdir, "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
    
    # to be deterministic
    cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.pidf in ["nccl","gloo"]:
        mp.spawn(dpp_train,args=(args.world_size,args,),nprocs=args.world_size,join=True)
    else:
        dpp_train(0, args.world_size, args)  

    endtime = time.time()
    interval = endtime - starttime
    print("elapsed time = {0:d}h {1:d}m {2:d}s".format(int(interval/3600), int((interval%3600)/60), int((interval%3600)%60)))