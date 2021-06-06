# -*- coding: utf-8 -*-

import argparse

def arguments():
	parser = argparse.ArgumentParser(description="training CNN models")
	# paths
	parser.add_argument("--train", '-t', default=None, type = str, help="path to training images")
	parser.add_argument("--val", '-val', default=None, type = str, help="path to validation images")
	parser.add_argument("--path2PHdir", '-pd', default="", type = str, help="path to precomputed PH npy files")
	parser.add_argument("--path2weight", '-pw', default="", type = str, help="path to trained weight")
	parser.add_argument("--output", '-o', default="result", type = str, help="path to output files")
	parser.add_argument('--resume', default='', type=str, help='path to checkpoint file to resume')
	parser.add_argument("--pidf", '-p', default=None, type = str, help="backend for parallelisation")
	parser.add_argument("--world_size", '-ws', default=2, type = int, help="number of GPUs to be used")
	# image generator
	parser.add_argument("--alpha_range", '-ar', default=[0.01,1], type = float, nargs=2, help="synthesised image frequency pattern")
	parser.add_argument("--beta_range", '-br', default=[0.5,2], type = float, nargs=2, help="synthesised image frequency pattern")
	parser.add_argument("--n_samples", '-n', default=50000, type = int, help="number of images generated in an epoch")
	parser.add_argument("--prob_binary", '-pb', default=0.5, type = float, help="probability of binarising the generated image")
	parser.add_argument("--prob_colour", '-pc', default=0.5, type = float, help="probability of generating colour images")
	# persistent homology (label) parameter
	parser.add_argument("--max_life", '-ml', default=50, type = int, help="maximum life time for PH regression")
	parser.add_argument("--max_birth", '-maxb', default=None, type = int, help="maximum birth time for PH regression")
	parser.add_argument('--min_birth', '-minb', type=int, default=None)
	parser.add_argument('--affine', '-aff', default=False, action='store_true', help='apply random affine transformation')
	parser.add_argument('--bandwidth', '-bd', type=int, default=1, help='bandwidth of label smoothing')
	parser.add_argument('--persImg_sigma', type=float, default=100, help='sigma for persistence image')
	parser.add_argument('--label_type', '-lt', default="PH_hist", choices=['raw','life_curve','PH_hist','persistence_image','class'], help='label type')
	parser.add_argument('--persistence_after_transform', '-pat', action="store_true", help='PH computation after applying transformation')
	parser.add_argument('--distance_transform', '-dt', action="store_true", default=True, help="apply distance transform")
	parser.add_argument('--gradient', '-g', action="store_true", default=False, help="apply gradient filter")
	# network settings
	parser.add_argument("--usenet", '-u', default="resnet50", type = str, help="network architecture")
	parser.add_argument("--epochs", '-e', default=90, type = int, help="number of training epochs")
	parser.add_argument("--numof_pretrained_classes", "-npc", default=-1, type = int, help="dimension of pre-training channels")
	parser.add_argument("--numof_classes", '-nc', default=100, type = int, help="num of fine-tuning classes")
	parser.add_argument("--numof_fclayer", default=4096, type = int, help="dimension of fc layer")
	# optimiser
	parser.add_argument("--lr_fine", '-lrp', default=0.01, type = float, help="initial learning rate for pretraining")
	parser.add_argument("--lr_pre", '-lrf', default=0.1, type = float, help="initial learning rate for finetuning")
	parser.add_argument("--momentum", default=0.9, type = float, help="momentum")
	parser.add_argument("--weight_decay", "-wd", default=1e-04, type = float, help="weight decay")
	parser.add_argument("--optimizer", "-op", default='sgd', type = str, help="optimiser and scheduler")
	# etc
	parser.add_argument("--start-epoch", default=1, type = int, help="starting epoch")
	parser.add_argument("--batch_size", '-b', default=96, type = int, help="batch size for training")
	parser.add_argument("--img_size", '-is', default=256, type = int, help="image size")
	parser.add_argument("--crop_size", '-cs', default=224, type = int, help="crop size")
	parser.add_argument("--save-interval", default=90, type = int, help="checkpointing frequency in epochs")
	parser.add_argument("--log-interval", default=100, type=int, help="logging frequency in iterations")
	parser.add_argument("--seed", '-s', default=1, type=int, help="random seed")
	parser.add_argument("--num_workers", '-nw', default=8, type = int, help="num of workers for dataloader")
	parser.add_argument("--output_training_images", '-oti', default=0, type = int, help="save sample training images to file")
	args = parser.parse_args()

	if args.max_birth is None:
		args.max_birth = args.max_life

	if args.min_birth is None:
		if args.gradient or not args.distance_transform:
			args.min_birth = 0
		else:
			args.min_birth = -args.max_birth

	return args
