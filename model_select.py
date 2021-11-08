# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn

import torchvision.models as models

# Not used
class DoubleHeadLinear(torch.nn.Module):
    def __init__(self, D_in, D_out1, D_out2):
        super(DoubleHeadLinear, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out1)
        self.linear2 = torch.nn.Linear(D_in, D_out2)

    def forward(self, x):
        y1 = self.linear1(x)
        y2 = self.linear2(x)
        return torch.cat([y1,y2], dim=-1)

# Model creation and weights initialisation
def model_select(args, outdim):
	param, pretrained = False, False
	train_from_scratch = False

	if (args.path2weight == "imagenet"):
		pretrained = True
		args.numof_pretrained_classes = 1000
		print("using imagenet pretrained model")
	else:
		if args.path2weight is None:
			print("weight file {} not found: training from scratch!\n".format(args.path2weight))
			args.numof_pretrained_classes = outdim
			train_from_scratch = True
		elif os.path.exists(args.path2weight):
			print ("using pretrained model : {}".format(args.path2weight))
			param = torch.load(args.path2weight, map_location=lambda storage, loc: storage)
			if param and "fc.bias" in param:
				args.numof_pretrained_classes = param["fc.bias"].shape[0]
			elif param and "classifier.bias" in param:
				args.numof_pretrained_classes = param["classifier.bias"].shape[0]
		else:
			print("Wrong weight file!")
			exit()
			
	arc = getattr(models, args.usenet)
	#model = eval(args.usenet+"(pretrained={}, num_classes={})".format(pretrained, args.numof_pretrained_classes))
	model = arc(pretrained=pretrained, num_classes=args.numof_pretrained_classes)

	# load weights from file
	if param:
		model.load_state_dict(param)

	# stop updating the encoder part (only the last fc will be trained)
	if args.freeze_encoder and param:
		for param in model.parameters():
			param.requires_grad = False
	
	if not train_from_scratch:
		# replace the last fc layer
		if "resne" in args.usenet:
			in_dim = model.fc.in_features
			last_layer = nn.Linear(in_dim, outdim)
			model.fc = last_layer
		elif "densenet" in args.usenet:
			in_dim = model.classifier.in_features
			last_layer = nn.Linear(in_dim, outdim)
			model.classifier = last_layer

	return model
