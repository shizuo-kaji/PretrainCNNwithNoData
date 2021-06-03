# -*- coding: utf-8 -*-
import time
import random
import os
import json
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
import torchvision.datasets as datasets

from arguments import arguments
from model_select import model_select
from train_val import train, validate

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
    train_transform = transforms.Compose([
                        transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.RandomResizedCrop(args.crop_size,scale=(0.7,1.0)),
                        #transforms.RandomCrop(args.crop_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = datasets.ImageFolder(args.train, transform=train_transform)
    if args.pidf in ["nccl","gloo"]:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=world_size,rank = rank)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                    num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn, sampler = train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.num_workers,
                                                pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
    
    # validation dataset
    if rank==0:
        test_transform = transforms.Compose([
                            transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.CenterCrop(args.crop_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        if not args.val:
            args.val=args.train.replace('train','val')
        test_dataset = datasets.ImageFolder(args.val,test_transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.num_workers,
                                                pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    # setup model and optimiser
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
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
        optimizer = optim.SGD(model.parameters(), lr=args.lr_fine, momentum=args.momentum, weight_decay=args.weight_decay)
    elif "adam" in args.optimizer:
        optimizer = optim.Adam(model.parameters(), lr=args.lr_fine)
        print("using Adam.")
    if "cos" in args.optimizer:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        print("using cosine annealing.")
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//3,2*args.epochs//3], gamma=0.1)

    # logger
    if rank==0:
        writer = SummaryWriter(log_dir=args.logdir)
    else:
        writer = None

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
    else:
        args.start_epoch=1

    # Training and Validation
    for epoch in range(1, args.epochs + 1):
        if args.pidf in ["nccl","gloo"]:
            train_sampler.set_epoch(epoch)
            loss, acc = train(args, model, rank, train_loader, optimizer, epoch, criterion)
        else:
            loss, acc = train(args, model, device, train_loader, optimizer, epoch, criterion)
        scheduler.step()
        if rank == 0:
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Acc/train", acc, epoch)
            validation_loss, validation_accuracy = validate(args, model, device, test_loader, criterion)
            writer.add_scalar("Loss/val", validation_loss, epoch)
            writer.add_scalar("Acc/val", validation_accuracy, epoch)
        if (epoch % args.save_interval == 0 or epoch == args.epochs) and rank==0:
            print("saving checkpoint...")
            saved_weight = os.path.join(args.output, "ft_"+args.usenet+"_epoch"+ str(epoch) +".pth")
            if args.world_size>1:
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            # Save checkpoint
            torch.save(model_state, saved_weight)
            checkpoint = "{}/{}_checkpoint.pth.tar".format(args.output, args.usenet)
            torch.save({'epoch': epoch + 1,
                        'state_dict': model_state,
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),}, checkpoint)

            model = model.to(device) #rank?

    if args.pidf in ["nccl","gloo"]:
        dist.destroy_process_group()

if __name__== "__main__":
    print(args)
    dtstr = dt.now().strftime('%Y_%m%d_%H%M')
    if "CIFAR100" in args.train:
        dtstr += "_C100"
    elif "omniglot" in args.train:        
        dtstr += "_omn"
    args.logdir = os.path.join(os.path.dirname(__file__),"runs/{}".format(dtstr))
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

    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), "GPUs available")

    starttime = time.time()
    if args.pidf in ["nccl","gloo"]:
        mp.spawn(dpp_train,args=(args.world_size,args,),nprocs=args.world_size,join=True)
    else:
        dpp_train(0, args.world_size, args)  

    endtime = time.time()
    interval = endtime - starttime
    print ("elapsed time = %dh %dm %ds" % (int(interval/3600),int((interval%3600)/60),int((interval%3600)%60)))
