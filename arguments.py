# -*- coding: utf-8 -*-

import argparse,sys,os,glob
import numpy as np

try:
    import torchvision.models as models
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
except:
    model_names = None

def arguments(mode="all"):
    parser = argparse.ArgumentParser(description="training CNN models")

    # common arguments
    parser.add_argument("--output", '-o', default="result", type = str, help="path to output files")
    parser.add_argument("--img_size", '-is', default=256, type = int, help="the size to which input images will be resized initially")
    parser.add_argument("--num_workers", '-j', default=4, type = int, help="num of workers for dataloader")

    if mode in ["all"]:
        # paths
        parser.add_argument("--train", '-t', default=None, type = str, help="path to training images")
        parser.add_argument("--val", '-val', default=None, type = str, help="path to validation images")
        parser.add_argument("--train_pt", '-t2', default=None, type = str, help="path to training images for pre-training (set to 'generate' for on-the-fly generation")
        parser.add_argument("--val_pt", '-v2', default=None, type = str, help="path to validation images for pre-training")
        parser.add_argument("--path2PHdir", '-pd', default=None, type = str, help="path to precomputed PH npy files")
        parser.add_argument("--path2weight", '-pw', default=None, type = str, help="path to trained weight (set to imagenet for ImageNet pretrained)")
        parser.add_argument("--cachedir", '-c', default=None, type = str, help="path for PH cache")
        parser.add_argument("--suffix", '-sf', default="", type = str, help="suffix to the output dir name (for making memo)")
        parser.add_argument('--resume', default='', type=str, help='path to checkpoint file to resume')

    # image generator
    if mode in ["all","random_image"]:
        parser.add_argument("--alpha_range", '-ar', default=[1,1], type = float, nargs=2, help="synthesised image frequency pattern")
        parser.add_argument("--beta_range", '-br', default=[1,2], type = float, nargs=2, help="synthesised image frequency pattern")
        parser.add_argument("--n_samples", '-n', default=400000, type = int, help="number of images generated for training")
        parser.add_argument("--n_samples_val", '-nv', default=1000, type = int, help="number of images generated for validation")
        parser.add_argument("--prob_binary", '-pb', default=0.5, type = float, help="probability of binarising the generated image")
        parser.add_argument("--prob_colour", '-pc', default=0.5, type = float, help="probability of generating colour images")

    # persistent homology (label) parameter
    if mode in ["all","PHdict"]:
        parser.add_argument("--max_life", '-ml', default=[50,50], type = int, nargs=2, help="maximum life time of each dimension for PH regression")
        parser.add_argument("--max_birth", '-maxb', default=None, type = int, nargs=2, help="maximum birth time of each dimension for PH regression")
        parser.add_argument('--min_birth', '-minb', type=int, default=None, nargs=2, help="minimum birth time of each dimension for PH regression")
        parser.add_argument('--affine', '-aff', default=False, action='store_true', help='apply random affine transformation')
        parser.add_argument('--bandwidth', '-bd', type=int, default=1, help='bandwidth of label smoothing for PH_hist')
        parser.add_argument('--persImg_sigma', '-ps', type=float, default=1, help='sigma for the gaussian kernel in persistence image')
        parser.add_argument('--persImg_power', '-pp', type=float, default=0.5, help='scaling for the vector')
        parser.add_argument('--persImg_weight', '-pn', type=float, default=1.0, help='weight for persistence weighting in persistence image')
        parser.add_argument('--num_landscapes', type=int, default=2, help='number of landscapes for persistence landscape')
        parser.add_argument('--label_type_pt', '-lt', default="persistence_image", choices=['raw','persistence_betticurve','persistence_histogram','persistence_image','persistence_landscape','class','grid'], help='label type for the pretraining task')
        parser.add_argument('--persistence_after_transform', '-pat', action="store_true", help='PH computation after applying transformation')
        parser.add_argument('--filtration', '-f', default='signed_distance', choices=[None,'distance','signed_distance','radial','radial_inv','upward','downward'], help="type of filtration")
        parser.add_argument('--gradient', '-g', action="store_true", default=False, help="apply gradient filter")

    if mode in ["all"]:
        # network settings
        parser.add_argument("--usenet", '-u', default="resnet50", type = str, choices=model_names, help="network architecture")
        parser.add_argument("--numof_dims_pt", '-nd', default=200, type = int, help="num of output dimensions for pretraining") #
        # optimiser
        parser.add_argument("--lr_fine", '-lrf', default=0.01, type = float, help="initial learning rate for pretraining") #0.01
        parser.add_argument("--lr_pre", '-lrp', default=0.1, type = float, help="initial learning rate for finetuning")
        parser.add_argument("--lr_drop", '-lrd', default=3, type = int, help="number of times lr drops by a factor of 10")
        parser.add_argument("--momentum", '-m', default=0.9, type = float, help="momentum")
        parser.add_argument("--weight_decay", "-wd", default=1e-4, type = float, help="weight decay") # 5e-4 for resnet18
        parser.add_argument("--optimizer", "-op", default='sgd', type = str, choices=["sgd","sgd_nesterov","sgd_cos","adam","adam_cos"], help="optimiser and scheduler")
        parser.add_argument('--freeze_encoder', '-fe', action="store_true", default=False, help="do not finetune encoder part")
        # etc
        parser.add_argument("--learning_mode", '-lm', default="combined", type = str, choices=["pretraining","finetuning","combined","alternating","simultaneous","evaluation"], help="How two tasks and datasets are used for learning")
        parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str, help='url used to set up distributed training')
        parser.add_argument('--dist-backend', '-p', default='nccl', type=str, help='distributed backend')
        parser.add_argument("--world_size", '-ws', default=None, type = int, help='number of nodes for distributed training')
        parser.add_argument("--epochs", '-e', default=90, type = int, help="number of training epochs") # 90 for resnet50, 200 for resnet18
        parser.add_argument("--start_epoch", default=1, type = int, help="starting epoch (useful on restart)")
        parser.add_argument("--batch_size", '-b', default=128, type = int, help="batch size for training")
        parser.add_argument("--img_size_pt", '-is2', default=228, type = int, help="the size to which input images for the pretraining task will be resized initially")
        parser.add_argument("--crop_size", '-cs', default=224, type = int, help="crop size")
        parser.add_argument("--crop_padding", '-cp', default=0, type = int, help="padding before cropping")
        parser.add_argument("--random_scale", '-rs', default=0, type = float, help="random scaling factor")
        parser.add_argument("--random_rotation", '-rr', default=0, type = float, help="random rotation degree")
        parser.add_argument("--aug-plus", '-ra', action="store_true",  help="augmentation of MoCo v2")
        parser.add_argument("--save_interval", default=-1, type = int, help="checkpointing frequency in epochs")
        parser.add_argument("--log_interval", default=200, type=int, help="logging frequency in iterations")
        parser.add_argument("--seed", '-s', default=-1, type=int, help="random seed")
        parser.add_argument("--output_training_images", '-oti', default=0, type = int, help="save sample training images to file")

    if mode in ["PHdict"]:
        parser.add_argument('target_dir',type=str)
        parser.add_argument('--num_bins', '-n', type=int, nargs="*", default=[50,50,50,50])
        parser.add_argument('--save_fig', '-sf', action="store_true", help="save graphs")

    args = parser.parse_args()

    if mode == "random_image":
        return(args)
    ############################################################
    # adjust default values
    # adjustment w.r.t. the possible minimum value for the image
    if args.max_birth is None:
        args.max_birth = [args.max_life[0],args.max_life[1]]

    if args.min_birth is None:
        if args.filtration=='signed_distance':
            args.min_birth = [-args.max_life[0],-args.max_life[1]]
        else:
            args.min_birth = [0,0]

    if mode == "PHdict":
        return(args)

    ##
    if args.save_interval <= 0:
        args.save_interval = args.epochs

    # set PH regression output dimensions
    if args.label_type_pt in ["persistence_betticurve","persistence_landscape"]:
        args.num_bins = [args.numof_dims_pt//2,args.numof_dims_pt-args.numof_dims_pt//2]
    elif args.label_type_pt == "persistence_image":
        args.num_bins = [args.numof_dims_pt//2,args.numof_dims_pt//2]
        for d in [0,1]:
            s = np.sqrt((args.max_birth[d]-args.min_birth[d])*args.max_life[d]/args.num_bins[d])
            p = int((args.max_birth[d]-args.min_birth[d])/s)
            q = int(args.max_life[d]/s)
            while p*q < args.num_bins[d]:
                s = max((args.max_birth[d]-args.min_birth[d])/(p+1), args.max_life[d]/(q+1))
                p = int((args.max_birth[d]-args.min_birth[d])/s)
                q = int(args.max_life[d]/s)
            #print(p,q,p*q)
    elif args.label_type_pt == "persistence_histogram":
        if args.gradient:
            b = args.numof_dims_pt//3
            args.num_bins = [b,b,1,args.numof_dims_pt-2*b-1]
        else:
            b = args.numof_dims_pt//4
            args.num_bins = [b,b,b,args.numof_dims_pt-3*b]

    # choose the latest weight file in the specified dir
    if args.path2weight is not None and os.path.isdir(args.path2weight):
        fns = sorted(glob.glob(os.path.join(args.path2weight,'*.pth')), key=lambda f: os.stat(f).st_mtime, reverse=True)
        args.path2weight = fns[0]


    # infer dataset name from dir name
    args.dataset_name, args.dataset_name_pt = None, None
    if args.train is not None:
        if not "train" in args.train and os.path.isdir(os.path.join(args.train,"train")):
            args.train = os.path.join(args.train,"train")
        dn = os.path.split(os.path.dirname(os.path.normpath(args.train)))[1] ## the second dir name from the leaf (that is, excluding "train")
        #print(os.path.dirname(os.path.normpath(args.train)))
        args.dataset_name = dn if dn else None
        if not args.val:
            for nm in ['test','val']:
                if os.path.isdir(args.train.replace('train',nm)):
                    args.val=args.train.replace('train',nm)

    if args.path2weight == "imagenet":
        args.dataset_name_pt = "IMN"
    elif args.train_pt == "generate":
        args.dataset_name_pt = "random{}kpb{}pc{}a{}-{}b{}-{}".format(args.n_samples//1000,args.prob_binary,args.prob_colour,args.alpha_range[0],args.alpha_range[1],args.beta_range[0],args.beta_range[1])
        if not args.val_pt:
            args.val_pt = "generate"
    elif args.train_pt is not None:
        if not "train" in args.train_pt and os.path.isdir(os.path.join(args.train_pt,"train")):
            args.train_pt = os.path.join(args.train_pt,"train")
        dn = os.path.split(os.path.dirname(os.path.normpath(args.train_pt)))[1]
        args.dataset_name_pt = dn if dn else None
        if not args.val_pt:
            for nm in ['test','val']:
                if os.path.isdir(args.train_pt.replace('train',nm)) and "train" in args.train_pt:
                    args.val_pt=args.train_pt.replace('train',nm)
    else:
        args.train_pt = args.train
        args.val_pt = args.val

    # set PHdir automatically
    grad = "grad" if args.gradient else ""
    if args.path2PHdir is None:
        if not args.persistence_after_transform and args.train_pt is not None:
            dn1,dn2 = os.path.split(os.path.dirname(os.path.normpath(args.train_pt)))
            phdn = os.path.join(dn1,"PH{}_{}_{}".format(grad,args.filtration,dn2))
            if os.path.isdir(phdn):
                args.path2PHdir = phdn
            else:
                args.path2PHdir = "on_the_fly"
        else:
            args.path2PHdir = "on_the_fly"
    elif args.persistence_after_transform:
        #args.persistence_after_transform = False
        print("loading pre-computed PH from path2PHdir and persistence_after_transform are incompatible!")
        exit()
    elif not (os.path.isdir(args.path2PHdir) or args.path2PHdir == "on_the_fly"):
        print("path2PHdir should be a directory name containing precomputed PH or 'on_the_fly' ")
        exit()

    #os.environ['TORCH_WARN_ONCE'] = 'YES'
    return args
