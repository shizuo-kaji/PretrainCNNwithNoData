# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn

import torchvision.models as models

def model_select(args):
	param, pretrained = False, False
	if (args.path2weight == "imagenet"):
		pretrained = True
		args.numof_pretrained_classes = 1000
		print("use imagenet pretrained model")
	else:
		if os.path.exists(args.path2weight):
			print ("use pretrained model : {}".format(args.path2weight))
			param = torch.load(args.path2weight, map_location=lambda storage, loc: storage)
		else:
			print ("weight file {} not found: train from scratch!\n".format(args.path2weight))

	if args.numof_pretrained_classes <= 0:
		if param and "fc.bias" in param:
			args.numof_pretrained_classes = param["fc.bias"].shape[0]
		elif param and "classifier.bias" in param:
			args.numof_pretrained_classes = param["classifier.bias"].shape[0]
		else:
			args.numof_pretrained_classes = args.numof_classes

	arc = getattr(models, args.usenet)
	#model = eval(args.usenet+"(pretrained={}, num_classes={})".format(pretrained, args.numof_pretrained_classes))
	model = arc(pretrained=pretrained, num_classes=args.numof_pretrained_classes)
	if param:
		model.load_state_dict(param)

	if "resne" in args.usenet:
		last_layer = nn.Linear(2048, args.numof_classes)
		model.fc = last_layer
	elif "densenet" in args.usenet:
		last_layer = nn.Linear(2208, args.numof_classes)
		model.classifier = last_layer

	return model
