# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn

import torchvision.models as models

def model_select(args):
	if args.numof_pretrained_classes <= 0:
		args.numof_pretrained_classes = args.numof_classes
		replace_fc = False
	else:
		replace_fc = True
	if (args.path2weight == "imagenet"):
		pretrained = True
		args.numof_pretrained_classes = 1000
	else:
		pretrained = False

	arc = getattr(models, args.usenet)
	#model = eval(args.usenet+"(pretrained={}, num_classes={})".format(pretrained, args.numof_pretrained_classes))
	model = arc(pretrained=pretrained, num_classes=args.numof_pretrained_classes)
	if os.path.exists(args.path2weight):
		print ("use pretrained model : {}".format(args.path2weight))
		param = torch.load(args.path2weight, map_location=lambda storage, loc: storage)
		model.load_state_dict(param)
	# ImageNet pre-trained model
	else:
		print ("{} not found: train from scratch!\n".format(args.path2weight))

	if replace_fc:
		if "resne" in args.usenet:
			last_layer = nn.Linear(2048, args.numof_classes)
			model.fc = last_layer
		elif "densenet" in args.usenet:
			last_layer = nn.Linear(2208, args.numof_classes)
			model.classifier = last_layer

	return model
