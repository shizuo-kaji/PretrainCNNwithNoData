# -*- coding: utf-8 -*-
import time
import random
import os,sys,shutil
import json
from datetime import datetime as dt
import socket

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
from dataset import DatasetFolderPH

from torch.utils.tensorboard import SummaryWriter

args = arguments()
val_rec = [] # record accuracy

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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.random_scale == 0:
        train_transform = transforms.Compose([
                            transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.RandomCrop(args.crop_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])
    else:
        train_transform = transforms.Compose([
                            transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.RandomResizedCrop(args.crop_size,scale=(1.0-3*args.random_scale,1.0)),
                            #transforms.RandomResizedCrop(args.crop_size,scale=(1.0-2*args.random_scale,1.0),ratio=(1.0-args.random_scale,1.0+args.random_scale)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])
    if args.learning_mode == "simultaneous":
        train_datasets = [DatasetFolderPH(args.train, transform=train_transform, args=args, PH_vect_dim=args.numof_classes2)]
        print(len(train_datasets[0]), "first train images loaded (used for both classification and PH regression).")
    else:
        train_datasets = [datasets.ImageFolder(args.train, transform=train_transform)]
        print(len(train_datasets[0]), "first train images loaded.")
    if args.train2 is not None and args.learning_mode != "simultaneous":
        train_datasets.append(DatasetFolderPH(args.train2, transform=train_transform, args=args, PH_vect_dim=args.numof_classes2))
        print(len(train_datasets[0]), len(train_datasets[1]), "first and second train images loaded.")
    print("number of classes: ", len(train_datasets[0].classes))

    if args.pidf in ["nccl","gloo"]:
        train_samplers = [torch.utils.data.distributed.DistributedSampler(train_datasets[0],num_replicas=world_size,rank = rank)]
        train_loaders = [torch.utils.data.DataLoader(train_datasets[0], batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn, sampler = train_samplers[0])]
        if args.train2 is not None and args.learning_mode != "simultaneous":
            train_samplers.append(torch.utils.data.distributed.DistributedSampler(train_datasets[1],num_replicas=world_size,rank = rank))
            train_loaders.append(torch.utils.data.DataLoader(train_datasets[1], batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn, sampler = train_samplers[1]))
    else:
        train_loaders = [torch.utils.data.DataLoader(dataset=train_datasets[0], batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers,pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)]
        if args.train2 is not None and args.learning_mode != "simultaneous":
            train_loaders.append(torch.utils.data.DataLoader(dataset=train_datasets[1], batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers,pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn))

    # validation dataset
    if rank==0:
        test_transform = transforms.Compose([
                            transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.CenterCrop(args.crop_size),
                            transforms.ToTensor(),
                            normalize])
        if args.learning_mode == "simultaneous":
            test_datasets = [DatasetFolderPH(args.val, transform=test_transform,args=args, PH_vect_dim=args.numof_classes2)]
        else:
            test_datasets = [datasets.ImageFolder(args.val,test_transform)]
        test_loaders = [torch.utils.data.DataLoader(dataset=test_datasets[0], batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.num_workers,
                                                pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)]
        print(len(test_datasets[0]), "first validation images loaded.")
        if args.train2 is not None and args.learning_mode != "simultaneous":
            test_datasets.append(DatasetFolderPH(args.val2, transform=test_transform,args=args, PH_vect_dim=args.numof_classes2))
            test_loaders.append(torch.utils.data.DataLoader(test_datasets[1], batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn))
            print(len(test_datasets[1]), "second validation images loaded.")
    # setup model and optimiser
    criterions = [nn.CrossEntropyLoss(reduction='mean').to(device)]
    if args.learning_mode != "single":
        parts = [(0,args.numof_classes),(args.numof_classes,args.numof_classes+args.numof_classes2)]
        criterions.append(nn.MSELoss(reduction='mean').to(device))
    else:
        parts = [None]

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
        print("LR drops at: ", [(i+1) * args.epochs//args.lr_drop for i in range(args.lr_drop-1)])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[(i+1) * args.epochs//args.lr_drop for i in range(args.lr_drop-1)], gamma=0.1)

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
        if args.learning_mode != "simultaneous":
            for i in range(len(criterions)):
                if args.pidf in ["nccl","gloo"]:
                    train_samplers[i].set_epoch(epoch)
                    loss, acc, _ = train(args, model, rank, train_loaders[i], optimizer, epoch, [criterions[i]], part=parts[i])
                else:
                    loss, acc, _ = train(args, model, device, train_loaders[i], optimizer, epoch, [criterions[i]], part=parts[i])
                if rank == 0:
                    validation_loss, validation_accuracy, _ = validate(args, model, device, test_loaders[i], [criterions[i]], part=parts[i])
                    if i==0:
                        val_rec.append(validation_accuracy)
                        writer.add_scalar("Loss/train", loss, epoch)
                        writer.add_scalar("Acc/train", acc, epoch)
                        writer.add_scalar("Loss/val", validation_loss, epoch)
                        writer.add_scalar("Acc/val", validation_accuracy, epoch)
                    else:
                        writer.add_scalar("Loss/train2", loss, epoch)
                        writer.add_scalar("Loss/val2", validation_loss, epoch)
        else: #simultaneous
            if args.pidf in ["nccl","gloo"]:
                train_samplers[i].set_epoch(epoch)
                loss, acc, loss2 = train(args, model, rank, train_loaders[0], optimizer, epoch, criterions, part=parts)
            else:
                loss, acc, loss2 = train(args, model, device, train_loaders[0], optimizer, epoch, criterions, part=parts)
            if rank == 0:
                validation_loss, validation_accuracy, validation_loss2 = validate(args, model, device, test_loaders[0], criterions, part=parts)
                writer.add_scalar("Loss/train", loss, epoch)
                writer.add_scalar("Acc/train", acc, epoch)
                writer.add_scalar("Loss/val", validation_loss, epoch)
                writer.add_scalar("Acc/val", validation_accuracy, epoch)
                writer.add_scalar("Loss/train2", loss2, epoch)
                writer.add_scalar("Loss/val2", validation_loss2, epoch)



        scheduler.step()
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
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), "GPUs available")
    else:
        args.world_size = 1

    if args.learning_mode != "single":
        args.world_size = 1
        print("world size set to 1")
    dtstr = dt.now().strftime('%Y_%m%d_%H%M')
    if "CIFAR100" in args.train:
        dtstr += "_C100"
    elif "omniglot" in args.train:        
        dtstr += "_OMN"
    elif "FGADR" in args.train:
        dtstr += "_FGADR"
    else:
        dtstr += "_rnd"
    dtstr += "_"+args.suffix
    args.logdir = os.path.join(os.path.dirname(__file__),"runs/{}/{}".format(socket.gethostname(),dtstr))
    args.output = os.path.join(os.path.expanduser(args.output), dtstr)
    print(args)

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

    starttime = time.time()
    if args.pidf in ["nccl","gloo"]:
        mp.spawn(dpp_train,args=(args.world_size,args,),nprocs=args.world_size,join=True)
    else:
        dpp_train(0, args.world_size, args)  

    endtime = time.time()
    interval = endtime - starttime
    print ("elapsed time = %dh %dm %ds" % (int(interval/3600),int((interval%3600)/60),int((interval%3600)%60)))
    print(args.logdir)
    with open(os.path.join(args.logdir,"args.txt"), 'w') as fh:
        fh.write(" ".join(sys.argv))
        fh.write(",".join([str(f) for f in val_rec]))

    try:
        source = os.path.join(os.path.dirname(args.path2weight),"args.json")
        shutil.copyfile(source, os.path.join(args.logdir,"args_pt.json"))
        shutil.copyfile(source, os.path.join(args.output,"args_pt.json"))
    except:
        pass 
