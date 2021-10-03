# -*- coding: utf-8 -*-

import argparse,sys,os,glob,torch

def arguments():
    parser = argparse.ArgumentParser(description="training CNN models")
    # paths
    parser.add_argument("--train", '-t', default=None, type = str, help="path to training images")
    parser.add_argument("--val", '-val', default=None, type = str, help="path to validation images")
    parser.add_argument("--train_pt", '-t2', default=None, type = str, help="path to training images for pre-training (set to generate for on-the-fly generation")
    parser.add_argument("--val_pt", '-v2', default=None, type = str, help="path to validation images for pre-training")
    parser.add_argument("--learning_mode", '-lm', default="combined", type = str, choices=["pretraining","finetuning","combined","alternating","simultaneous"], help="How two tasks and datasets are used for learning")
    parser.add_argument("--path2PHdir", '-pd', default="", type = str, help="path to precomputed PH npy files")
    parser.add_argument("--path2weight", '-pw', default=None, type = str, help="path to trained weight (set to imagenet for ImageNet pretrained)")
    parser.add_argument("--output", '-o', default="result", type = str, help="path to output files")
    parser.add_argument("--suffix", '-sf', default="", type = str, help="suffix to the output dir name (for making memo)")
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint file to resume')
    parser.add_argument("--pidf", '-p', default="DPP", type = str, help="backend for parallelisation")
    parser.add_argument("--world_size", '-ws', default=1, type = int, help="number of GPUs to be used")

    # image generator
    parser.add_argument("--alpha_range", '-ar', default=[0.01,1], type = float, nargs=2, help="synthesised image frequency pattern")
    parser.add_argument("--beta_range", '-br', default=[0.5,2], type = float, nargs=2, help="synthesised image frequency pattern")
    parser.add_argument("--n_samples", '-n', default=50000, type = int, help="number of images generated in an epoch")
    parser.add_argument("--prob_binary", '-pb', default=0.5, type = float, help="probability of binarising the generated image")
    parser.add_argument("--prob_colour", '-pc', default=0.5, type = float, help="probability of generating colour images")
    # persistent homology (label) parameter
    parser.add_argument("--max_life", '-ml', default=[40,40], type = int, nargs=2, help="maximum life time of each dimension for PH regression")
    parser.add_argument("--max_birth", '-maxb', default=None, type = int, nargs=2, help="maximum birth time of each dimension for PH regression")
    parser.add_argument('--min_birth', '-minb', type=int, default=None, nargs=2, help="minimum birth time of each dimension for PH regression")
    parser.add_argument('--affine', '-aff', default=False, action='store_true', help='apply random affine transformation')
    parser.add_argument('--bandwidth', '-bd', type=int, default=1, help='bandwidth of label smoothing')
    parser.add_argument('--persImg_sigma', type=float, default=100, help='sigma for persistence image')
    parser.add_argument('--label_type_pt', '-lt', default="PH_hist", choices=['raw','life_curve','PH_hist','persistence_image','landscape','class'], help='label type for the pretraining task')
    parser.add_argument('--persistence_after_transform', '-pat', action="store_true", help='PH computation after applying transformation')
    parser.add_argument('--distance_transform', '-dt', action="store_true", default=True, help="apply distance transform")
    parser.add_argument('--gradient', '-g', action="store_true", default=False, help="apply gradient filter")
    parser.add_argument('--greyscale', '-gr', action="store_true", default=False, help="make loaded image greyscale")

    # network settings
    parser.add_argument("--usenet", '-u', default="resnet50", type = str, help="network architecture")
    parser.add_argument("--epochs", '-e', default=90, type = int, help="number of training epochs") # 90
    #parser.add_argument("--numof_classes", '-nc', default=100, type = int, help="num of dimensions for the main task")
    parser.add_argument("--numof_dims_pt", '-nd', default=148, type = int, help="num of output dimensions for pretraining")
    # optimiser
    parser.add_argument("--lr_fine", '-lrf', default=0.01, type = float, help="initial learning rate for pretraining") #0.01
    parser.add_argument("--lr_pre", '-lrp', default=0.1, type = float, help="initial learning rate for finetuning")
    parser.add_argument("--lr_drop", '-lrd', default=3, type = int, help="number of times lr drops by a factor of 10")
    parser.add_argument("--momentum", '-m', default=0.9, type = float, help="momentum")
    parser.add_argument("--weight_decay", "-wd", default=1e-4, type = float, help="weight decay") # 5e-4 for resnet18
    parser.add_argument("--optimizer", "-op", default='sgd', type = str, choices=["sgd","sgd_nesterov","sgd_cos","adam","adam_cos"], help="optimiser and scheduler")
    parser.add_argument('--freeze_encoder', '-fe', action="store_true", default=False, help="do not finetune encoder part")
    # etc
    parser.add_argument("--start_epoch", default=1, type = int, help="starting epoch")
    parser.add_argument("--batch_size", '-b', default=128, type = int, help="batch size for training")
    parser.add_argument("--img_size", '-is', default=252, type = int, help="input images will be resized initially") #256
    parser.add_argument("--img_size_pt", '-is2', default=228, type = int, help="input images for the pretraining task will be resized initially")
    parser.add_argument("--crop_size", '-cs', default=224, type = int, help="crop size")
    parser.add_argument("--crop_padding", '-cp', default=0, type = int, help="padding before cropping")
    parser.add_argument("--random_scale", '-rs', default=0, type = float, help="random scaling factor")
    parser.add_argument("--save_interval", default=-1, type = int, help="checkpointing frequency in epochs")
    parser.add_argument("--log_interval", default=200, type=int, help="logging frequency in iterations")
    parser.add_argument("--seed", '-s', default=0, type=int, help="random seed")
    parser.add_argument("--num_workers", '-nw', default=4, type = int, help="num of workers for dataloader")
    parser.add_argument("--output_training_images", '-oti', default=0, type = int, help="save sample training images to file")
    args = parser.parse_args()

    # adjust default values
    if torch.cuda.device_count() <= 1:
        #print(torch.cuda.device_count(), "GPUs available")
        args.world_size = 1

    if args.save_interval <= 0:
        args.save_interval = args.epochs

    # adjustment w.r.t. the possible minimum value for the image
    if args.max_birth is None:
        args.max_birth = [args.max_life[0],args.max_life[1]]

    if args.min_birth is None:
        if not args.distance_transform or args.gradient:
            args.min_birth = [0,0]
        else:
            args.min_birth = [-args.max_life[0],-args.max_life[1]]

    # if PH is not precomputed, compute PH after transformation
    if args.path2PHdir is None:
        args.persistence_after_transform = True

    # choose the latest weight file in the specified dir
    if args.path2weight is not None and os.path.isdir(args.path2weight):
        fns = sorted(glob.glob(os.path.join(args.path2weight,'*.pth')), key=lambda f: os.stat(f).st_mtime, reverse=True)
        args.path2weight = fns[0]


    # infer dataset name from dir name
    args.dataset_name, args.dataset_name_pt = None, None
    if args.train is not None:
        if not "train" in args.train and os.path.isdir(os.path.join(args.train,"train")):
            args.train = os.path.join(args.train,"train")
        dn = os.path.split(os.path.dirname(os.path.normpath(args.train)))[1]
        #print(os.path.dirname(os.path.normpath(args.train)))
        args.dataset_name = dn if dn else None
        if not args.val:
            for nm in ['val','test']:
                if os.path.isdir(args.train.replace('train',nm)):
                    args.val=args.train.replace('train',nm)

    if args.path2weight == "imagenet":
        args.dataset_name_pt = "IMN"
    elif args.train_pt == "generate":
        args.dataset_name_pt = "GEN"
    elif args.train_pt is not None:
        if not "train" in args.train_pt and os.path.isdir(os.path.join(args.train_pt,"train")):
            args.train_pt = os.path.join(args.train_pt,"train")
        dn = os.path.split(os.path.dirname(os.path.normpath(args.train_pt)))[1]
        args.dataset_name_pt = dn if dn else None
        if not args.val_pt:
            for nm in ['val','test']:
                if os.path.isdir(args.train_pt.replace('train',nm)):
                    args.val_pt=args.train_pt.replace('train',nm)
    else:
        args.train_pt = args.train
        args.val_pt = args.val


    if args.path2PHdir and args.persistence_after_transform:
        #args.persistence_after_transform = False
        print("loading pre-computed PH from path2PHdir and ersistence_after_transform are incompatible!")
        exit()


    #os.environ['TORCH_WARN_ONCE'] = 'YES'
    return args
