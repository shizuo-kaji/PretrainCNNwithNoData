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
	parser.add_argument('--persistence', '-ph', default=None, type=str, choices=[None, 'pre', 'post'], help='PH computation pre or post transformation')
	parser.add_argument('--precomputed', '-pc', default=False, action='store_true', help='pre-computed persistence histogram/image')
	# image generator
	parser.add_argument("--alpha_range", '-ar', default=[0.01,1], type = float, nargs=2, help="synthesised image frequency pattern")
	parser.add_argument("--beta_range", '-br', default=[0.5,2], type = float, nargs=2, help="synthesised image frequency pattern")
	parser.add_argument("--n_samples", '-n', default=50000, type = int, help="number of images generated in an epoch")
	parser.add_argument("--prob_binary", '-pb', default=1.0, type = float, help="probability of binarising the generated image")
	parser.add_argument("--max_life", '-ml', default=50, type = int, help="life time for PH regression")
	parser.add_argument('--affine', '-aff', default=False, action='store_true', help='apply random affine transformation')
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
	parser.add_argument("--img_size", default=256, type = int, help="image size")
	parser.add_argument("--crop_size", default=224, type = int, help="crop size")
	parser.add_argument("--save-interval", '-si', default=90, type = int, help="checkpointing frequency in epochs")
	parser.add_argument("--log-interval", default=100, type=int, help="logging frequency in iterations")
	parser.add_argument("--seed", default=1, type=int, help="random seed")
	parser.add_argument("--num_workers", '-nw', default=8, type = int, help="num of workers for dataloader")
	parser.add_argument("--output_training_images", '-oti', default=0, type = int, help="save sample training images to file")
	args = parser.parse_args()
	return args
