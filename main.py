# -*- coding: utf-8 -*-

import os,random,glob,time,shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json,gc
import numpy as np
from PIL import Image
from datetime import datetime as dt
import socket
import subprocess,sys

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

class Experiment:
    def __init__(self, log_dir: str):
        self.logger = None
        self.log_dir = log_dir
        self.val_rec = []

    def setup_logger(self):
        if self.logger is None:
            self.logger = SummaryWriter(self.log_dir)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def dpp_train(rank, world_size, args, mode="", exp=None):
    # setup parallelisation environment
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # create default process group
    if args.dist_backend == "nccl":
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '8888'
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        args.batch_size = int(args.batch_size / args.world_size)
        args.num_workers = int((args.num_workers + args.world_size - 1) / args.world_size)
    elif args.dist_backend == "gloo":
        pidf = os.path.join(args.output,"temp") ## check write permission!
        if os.path.isfile(pidf):
            os.remove(pidf)
        pid = "file:///{}".format(pidf)
        dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=pid)
        torch.cuda.set_device(rank)
        args.batch_size = int(args.batch_size / args.world_size)
        args.num_workers = int((args.num_workers + args.world_size - 1) / args.world_size)
    elif args.dist_backend is None:
        pass
    else:
        print("Unknown parallelisation backend")
        exit()
    if rank == 0:
        print("Phase: ", mode)
        print("paralell trainer: ",args.dist_backend)
        exp.setup_logger()

    # training dataset
    if mode=="pretraining":
        train_data_path = args.train_pt
        val_data_path = args.val_pt
    elif mode=="evaluation":
        train_data_path = args.val
        val_data_path = args.val
        args.epochs=1
    else:
        train_data_path = args.train
        val_data_path = args.val

    # training transform
    img_size = args.img_size_pt if mode=="pretraining" else args.img_size
    if train_data_path=="generate" or "random" in train_data_path:
        normalize = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.32,0.32,0.32])
    else:
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([])
        if args.random_scale > 0:
            train_transform.transforms.append(transforms.RandomResizedCrop(args.crop_size,scale=(1.0-args.random_scale,1.0)))
        elif img_size > args.crop_size:
            train_transform.transforms.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR))
        if args.random_rotation > 0:
            train_transform.transforms.append(transforms.RandomRotation(args.random_rotation))
        if args.affine:
            train_transform.transforms.append(transforms.RandomAffine(degrees=(-180,180), scale=(0.5,2), shear=(-100,100,-100,100)))
        train_transform.transforms.append(transforms.RandomCrop(args.crop_size, padding=args.crop_padding))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)

    # dataset loading
    if args.learning_mode == "simultaneous": # use a single dataset but with two tasks: classification and PH
        train_datasets = [DatasetFolderPH(train_data_path, transform=train_transform, args=args)]
        if rank == 0:
            print(len(train_datasets[0]), "first train images loaded (used for both classification and PH regression).")
        numof_classes = train_datasets[0].n_classes
        outdim = numof_classes+args.numof_dims_pt
        parts = [(0,numof_classes),(numof_classes,outdim)]
        criterions=[nn.CrossEntropyLoss(reduction='mean').to(rank),nn.MSELoss(reduction='mean').to(rank)]
    else:
        if args.label_type_pt == "class" or mode in ["finetuning","evaluation"]:
            train_datasets = [datasets.ImageFolder(train_data_path, transform=train_transform)]
            outdim = len(train_datasets[0].classes)
            if rank == 0:
                print(len(train_datasets[0]), "first train images loaded from ", train_data_path)
                print("number of classes: ", outdim)
            criterions = [nn.CrossEntropyLoss(reduction='mean').to(rank)]
        else: ## PH related tasks
            if rank == 0:
                if os.path.isdir(args.path2PHdir):
                    print("PH will be loaded from: ",args.path2PHdir)
                # caching
                if args.cachedir is not None:
                    if os.path.exists(args.cachedir):
                        print("clearning cache in {}".format(args.cachedir))
                        shutil. rmtree(args.cachedir)
                    os.makedirs(args.cachedir,exist_ok=True)
                    if train_data_path=="generate":
                        os.makedirs(os.path.join(args.cachedir,'train'),exist_ok=True)
                        os.makedirs(os.path.join(args.cachedir,'val'),exist_ok=True)
                    print("PH computation will be cached in ",args.cachedir)
            if train_data_path=="generate":
                imdir = os.path.join(args.cachedir,'train') if args.cachedir is not None else "train"
                train_datasets = [DatasetFolderPH(root=imdir, transform=train_transform, generate_on_the_fly=True, args=args)]
                if rank == 0:
                    print(len(train_datasets[0]), "training images are generated on the fly.")
            else:
                train_datasets = [DatasetFolderPH(train_data_path, transform=train_transform, args=args)]
                if rank == 0:
                    print(len(train_datasets[0]), "first train images loaded from ", train_data_path)
            criterions = [nn.MSELoss(reduction='mean').to(rank)]
            outdim = args.numof_dims_pt
        # second dataset
        if args.learning_mode == "alternating":
            train_datasets.append(DatasetFolderPH(args.train_pt, transform=train_transform, args=args, PH_vect_dim=args.numof_dims_pt))
            if rank == 0:
                print(len(train_datasets[0]), len(train_datasets[1]), "first and second train images loaded.")
            numof_classes = train_datasets[0].n_classes
            outdim = numof_classes+args.numof_dims_pt
            parts = [(0,numof_classes),(numof_classes,outdim)]
            criterions.append(nn.MSELoss(reduction='mean').to(rank))
        else:
            parts = [None]

    # save transformed sample training images to file
    for i in range(args.output_training_images):
        img = (train_datasets[0][i][0]).numpy()
        img = (255*(img-img.min())/np.ptp(img)).astype(np.uint8).transpose(1,2,0)
        Image.fromarray(img).save(os.path.join(args.output,"{:0>5}.jpg".format(i)))

    # data sampler
    if args.dist_backend in ["nccl","gloo"]:
        train_samplers = [torch.utils.data.distributed.DistributedSampler(train_datasets[0],num_replicas=world_size,rank = rank)]
        train_loaders = [torch.utils.data.DataLoader(train_datasets[0], batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn, sampler = train_samplers[0])]
        if args.learning_mode == "alternating":
            train_samplers.append(torch.utils.data.distributed.DistributedSampler(train_datasets[1],num_replicas=world_size,rank = rank))
            train_loaders.append(torch.utils.data.DataLoader(train_datasets[1], batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn, sampler = train_samplers[1]))
    else:
        train_loaders = [torch.utils.data.DataLoader(dataset=train_datasets[0], batch_size=args.batch_size,shuffle=True,
                                            num_workers=args.num_workers,pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)]
        if args.learning_mode == "alternating":
            train_loaders.append(torch.utils.data.DataLoader(dataset=train_datasets[1], batch_size=args.batch_size,shuffle=True,
                                            num_workers=args.num_workers,pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn))


    # validation dataset
    if val_data_path is not None and rank==0:
        test_transform = transforms.Compose([])
        if img_size > args.crop_size:
            test_transform.transforms.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR))
        test_transform.transforms.append(transforms.CenterCrop(args.crop_size))
        test_transform.transforms.append(transforms.ToTensor())
        test_transform.transforms.append(normalize)
        if args.label_type_pt == "class" or mode in ["finetuning","evaluation"]:
            test_datasets = [datasets.ImageFolder(val_data_path, transform=test_transform)]
            if rank == 0:
                print(len(test_datasets[0]), f"first validation images loaded from {val_data_path}.")
        else:
            if val_data_path=="generate":
                imdir = os.path.join(args.cachedir,'val') if args.cachedir is not None else ""
                test_datasets = [DatasetFolderPH(root=imdir, transform=train_transform, generate_on_the_fly=True, args=args)]
                if rank == 0:
                    print(len(test_datasets[0]), "validation images are generated on the fly.")
            else:
                test_datasets = [DatasetFolderPH(val_data_path, transform=test_transform,args=args)]
                if rank == 0:
                    print(len(test_datasets[0]), f"first validation images loaded from {val_data_path}.")

        test_loaders = [torch.utils.data.DataLoader(test_datasets[0], batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)]
        if args.learning_mode == "alternating":
            test_datasets.append(DatasetFolderPH(args.val_pt, transform=test_transform,args=args, PH_vect_dim=args.numof_dims_pt))
            test_loaders.append(torch.utils.data.DataLoader(test_datasets[1], batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn))
            if rank == 0:
                print(len(test_datasets[1]), "second validation images loaded.")

    # setup model and optimiser
    if args.learning_mode == "evaluation":
        model = model_select(args, outdim, renew_fc=False)
    else:
        model = model_select(args, outdim)

    if args.world_size != 1:
        if args.dist_backend in ["nccl","gloo"]:
            model = DDP(model.to(rank), device_ids=[rank],output_device=rank)
        else:
            model = nn.DataParallel(model)
            model = model.to(device)
    else:
        model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print("NN output dimension: ", outdim)
        print('The number of parameters of model is', num_params)

    lr = args.lr_fine if mode=="finetuning" else args.lr_pre
    if "sgd" in args.optimizer:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, nesterov=("nes" in args.optimizer), weight_decay=args.weight_decay)
    elif "adam" in args.optimizer:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print("using Adam.")

    if "cos" in args.optimizer:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        print("using cosine annealing.")
    else:
        if args.epochs == 200: # TEMP: resnet18
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        else:
            if rank == 0:
                print("LR drops at: ", [(i+1) * args.epochs//args.lr_drop for i in range(args.lr_drop-1)])
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[(i+1) * args.epochs//args.lr_drop for i in range(args.lr_drop-1)], gamma=0.1)

    # checkpointing
    if args.resume:
        if args.dist_backend in ["nccl","gloo"]:
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
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.learning_mode != "simultaneous":
            for i in range(len(criterions)):
                if args.learning_mode == "evaluation":
                    loss, acc = 0,0
                else:
                    if args.dist_backend in ["nccl","gloo"]:
                        train_samplers[i].set_epoch(epoch)
                        loss, acc, _ = train(args, model, rank, train_loaders[i], optimizer, epoch, [criterions[i]], part=parts[i], quiet=(rank!=0))
                    else:
                        loss, acc, _ = train(args, model, device, train_loaders[i], optimizer, epoch, [criterions[i]], part=parts[i])
                if rank == 0 and val_data_path is not None:
                    validation_loss, validation_accuracy, _ = validate(args, model, device, test_loaders[i], [criterions[i]], part=parts[i])
                    if i==0 and mode != "pretraining":
                        exp.val_rec.append(validation_accuracy.item())
                        exp.logger.add_scalar("Loss/train", loss, epoch)
                        exp.logger.add_scalar("Acc/train", acc, epoch)
                        exp.logger.add_scalar("Loss/val", validation_loss, epoch)
                        exp.logger.add_scalar("Acc/val", validation_accuracy, epoch)
                    else: # stat for pretraining task
                        exp.logger.add_scalar("Loss/train2", loss, epoch)
                        exp.logger.add_scalar("Loss/val2", validation_loss, epoch)
        else: #simultaneous
            if args.dist_backend in ["nccl","gloo"]:
                train_samplers[0].set_epoch(epoch)
                loss, acc, loss2 = train(args, model, rank, train_loaders[0], optimizer, epoch, criterions, part=parts, quiet=(rank!=0))
            else:
                loss, acc, loss2 = train(args, model, device, train_loaders[0], optimizer, epoch, criterions, part=parts)
            if rank == 0 and val_data_path is not None:
                validation_loss, validation_accuracy, validation_loss2 = validate(args, model, device, test_loaders[0], criterions, part=parts)
                exp.val_rec.append(validation_accuracy.item())
                exp.logger.add_scalar("Loss/train", loss, epoch)
                exp.logger.add_scalar("Acc/train", acc, epoch)
                exp.logger.add_scalar("Loss/val", validation_loss, epoch)
                exp.logger.add_scalar("Acc/val", validation_accuracy, epoch)
                exp.logger.add_scalar("Loss/train2", loss2, epoch)
                exp.logger.add_scalar("Loss/val2", validation_loss2, epoch)

        scheduler.step()
        if ((args.save_interval>0 and epoch % args.save_interval == 0) or epoch == args.epochs) and rank==0 and args.learning_mode != "evaluation":
            print("saving checkpoint...")
            if args.world_size>1:
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            # Save checkpoint
            model_type = "pt" if mode=="pretraining" else args.learning_mode
            saved_weight = "{}/{}_{}_epoch{}.pth".format(args.output, args.usenet, model_type, epoch)
            torch.save(model_state, saved_weight)
            checkpoint = "{}/checkpoint_epoch{}.pth.tar".format(args.output, epoch)
            torch.save({'epoch': epoch + 1,
                        'state_dict': model_state,
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),}, checkpoint)

    if args.dist_backend in ["nccl","gloo"]:
        dist.destroy_process_group()

    model_type = "pt" if mode=="pretraining" else args.learning_mode
    with open(os.path.join(args.logdir,"args_{}.txt".format(model_type)), 'w') as fh:
        fh.write(" ".join(sys.argv))
        fh.write("\n")
        fh.write("\n".join([str(f) for f in exp.val_rec]))

#####################################
if __name__== "__main__":

    # number of GPUs
    ngpus_per_node = torch.cuda.device_count()
    if args.world_size is None:
        args.world_size = max(ngpus_per_node,1)
    if ngpus_per_node <= 1 or args.learning_mode == "evaluation":
        #print(torch.cuda.device_count(), "GPUs available")
        args.world_size = 1
        args.dist_backend = None

    # create directories for log and output
    dtstr = dt.now().strftime('%Y_%m%d_%H%M')
    if args.learning_mode == "finetuning":
        dtstr += "_finetuning"
        if args.path2weight is None:
            dtstr += "_Scratch"
        elif "persistence_image" in args.path2weight:
            dtstr += "_PH-PI"
        elif "persistence_landscape" in args.path2weight:
            dtstr += "_PH-LS"
        elif "persistence_betticurve" in args.path2weight:
            dtstr += "_PH-BC"
        elif "persistence_histogram" in args.path2weight:
            dtstr += "_PH-HS"
        elif "Fractal" in args.path2weight:
            dtstr += "_FDB"
        elif "imagenet" in args.path2weight:
            dtstr += "_IMN"
        elif "class" in args.path2weight:
            dtstr += "_Label"
    else:
        grd ="grad_" if args.gradient else ""
        if args.label_type_pt != 'class':
            dtstr += '_{}{}_ml{}_n{}'.format(grd,args.label_type_pt,args.max_life[0],args.numof_dims_pt)
        else:
            dtstr += '_{}{}'.format(grd,args.label_type_pt)
    if args.dataset_name_pt:
        dtstr += "_"+args.dataset_name_pt
    if args.dataset_name:
        dtstr += "_ft-"+args.dataset_name
    if args.suffix:
        dtstr += "_"+args.suffix

    args.logdir = os.path.join(os.path.dirname(__file__),"runs/{}/{}".format(socket.gethostname(),dtstr))
    exp = Experiment(args.logdir)
    args.output = os.path.join(os.path.expanduser(args.output), dtstr)  #.replace("result","weight")
    print(args)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.output, "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
    with open(os.path.join(args.logdir, "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    cudnn.benchmark = True
    # to be deterministic
    if args.seed >= 0:
        cudnn.deterministic = True
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # pretraining and finetuing iterations
    if args.learning_mode=="combined":
        modes = ["pretraining","finetuning"]
    else:
        modes = [args.learning_mode]

    for mode in modes:
        starttime = time.time()
        if args.dist_backend in ["nccl","gloo"]:
            mp.spawn(dpp_train,args=(args.world_size,args,mode,exp),nprocs=args.world_size,join=True)
        else:
            dpp_train(0, args.world_size, args, mode,exp)
        endtime = time.time()
        interval = endtime - starttime
        print("{0} elapsed time = {1:d}h {2:d}m {3:d}s".format(mode, int(interval/3600), int((interval%3600)/60), int((interval%3600)%60)))
        print(args.logdir)
        # clear memory
        gc.collect()
        torch.cuda.empty_cache()
        # set weight file
        fns = sorted(glob.glob(os.path.join(args.output,'*.pth')), key=lambda f: os.stat(f).st_mtime, reverse=True)
        if len(fns)>0:
            args.path2weight = fns[0]

    print(args)
